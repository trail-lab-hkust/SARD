import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx


ALLOWED_ACTIONS = {
    "select_candidate_strategy",
    "rewrite_library_strategy",
    "create_new_strategy",
}

ALLOWED_DIAGNOSIS_TYPES = {
    "strategy_description_bad",
    "wrong_strategy_selected",
    "strategy_missing",
}

REQUIRED_STRATEGY_KEYS = {
    "objective",
    "evidence_to_promote",
    "evidence_to_demote",
    "ranking_procedure",
    "tie_breaking_rule",
    "fallback_behavior",
}


@dataclass
class RepairLLMResult:
    success: bool
    decision: dict[str, Any] | None = None
    raw_response: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    latency_s: float = 0.0
    retry_count: int = 0
    attempts: list[dict[str, Any]] | None = None


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return config.get(key, default)


def _chat_completions_url(base_url: str) -> str:
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise ValueError("guided repair llm base_url is required.")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response does not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])

    raise ValueError("LLM response JSON object is incomplete.")


def _validate_decision(decision: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, dict):
        raise ValueError("LLM decision must be a JSON object.")

    action = decision.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid action: {action}.")

    diagnosis_type = decision.get("diagnosis_type")
    if diagnosis_type not in ALLOWED_DIAGNOSIS_TYPES:
        raise ValueError(f"Invalid diagnosis_type: {diagnosis_type}.")

    current_strategy = request.get("current_strategy", {})
    candidates = request.get("candidate_strategy_library") or request.get("candidate_strategies") or []
    candidate_map = {
        candidate.get("candidate_key", candidate.get("strategy_id")): candidate
        for candidate in candidates
        if isinstance(candidate, dict)
    }
    selected_id = decision.get("selected_candidate_key", decision.get("selected_strategy_id"))

    if action == "select_candidate_strategy":
        selected = candidate_map.get(selected_id)
        if selected is None:
            raise ValueError(f"selected_candidate_key {selected_id!r} is not in candidate_strategy_library.")
        if selected.get("status") != "candidate":
            raise ValueError("select_candidate_strategy may only select a candidate strategy that has not been tried yet.")
        selected_strategy = selected.get("strategy") or selected.get("strategy_prompt_structured")
        if not isinstance(selected_strategy, dict):
            raise ValueError("Selected candidate does not contain a strategy card.")
        decision["new_strategy"] = selected_strategy

    elif action == "rewrite_library_strategy":
        selected = candidate_map.get(selected_id)
        if selected is None:
            raise ValueError(f"selected_candidate_key {selected_id!r} is not in candidate_strategy_library.")
        new_strategy = decision.get("new_strategy")
        if not isinstance(new_strategy, dict):
            raise ValueError("rewrite_library_strategy requires new_strategy to be a JSON object.")
        if any(
            isinstance(candidate, dict)
            and new_strategy == (candidate.get("strategy") or candidate.get("strategy_prompt_structured"))
            for candidate in candidates
        ):
            raise ValueError("rewrite_library_strategy requires a genuinely rewritten strategy not already present in the library.")

    elif action == "create_new_strategy":
        if selected_id is not None:
            raise ValueError("create_new_strategy requires selected_candidate_key to be null.")
        new_strategy = decision.get("new_strategy")
        if not isinstance(new_strategy, dict):
            raise ValueError("create_new_strategy requires new_strategy to be a JSON object.")
        if any(
            isinstance(candidate, dict)
            and new_strategy == (candidate.get("strategy") or candidate.get("strategy_prompt_structured"))
            for candidate in candidates
        ):
            raise ValueError("create_new_strategy requires a strategy not already present in the library.")

    new_strategy = decision.get("new_strategy")
    if not isinstance(new_strategy, dict):
        raise ValueError("new_strategy must resolve to a JSON object.")

    required_keys = set(REQUIRED_STRATEGY_KEYS)
    if (
        isinstance(current_strategy, dict)
        and "retention_rule" in current_strategy
        or any(
            isinstance(candidate, dict)
            and "retention_rule" in (candidate.get("strategy") or candidate.get("strategy_prompt_structured") or {})
            for candidate in candidates
        )
    ):
        required_keys.add("retention_rule")
    missing = [key for key in required_keys if key not in new_strategy]
    if missing:
        raise ValueError(f"new_strategy missing required keys: {missing}.")

    confidence = decision.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    decision["confidence"] = min(1.0, max(0.0, confidence))
    return decision


def _build_messages(request: dict[str, Any]) -> list[dict[str, str]]:
    required_output_format = (
        "<think>...</think> and <answer>...</answer>"
        if request.get("data_source") == "sard_reward_think"
        else "<thinking>...</thinking> and <answer>...</answer>"
    )
    system = (
        "You are a strategy repair module for a reinforcement learning ranking system. "
        "You are not solving the ranking problem. Your job is to inspect the ranking problem, "
        "the current strategy that failed, the student's best failed rollout under that strategy, "
        "and a library of candidate strategies, then produce exactly one improved strategy for the next student rollout. "
        "Do not output a passage ranking. Do not reveal or guess final passage IDs. "
        "Return only valid JSON matching the required schema."
    )

    user = {
        "task": "repair_ranking_strategy",
        "instruction": (
            {
                "goal": "Produce one improved strategy card for the next student rollout.",
                "allowed_actions": [
                    "select_candidate_strategy",
                    "rewrite_library_strategy",
                    "create_new_strategy",
                ],
                "action_meanings": {
                    "select_candidate_strategy": (
                        "Choose one unused library strategy whose status is candidate and use it unchanged next. "
                        "Use this only when an untried candidate already looks suitable; do not provide a rewritten strategy card."
                    ),
                    "rewrite_library_strategy": (
                        "Choose any library strategy as a source, including current, chosen, or candidate, then produce a genuinely "
                        "rewritten strategy card that is different from every strategy already in the library."
                    ),
                    "create_new_strategy": (
                        "None of the existing library strategies is suitable as a direct choice or rewrite source. Create a genuinely new strategy."
                    ),
                },
                "constraints": [
                    "Return JSON only.",
                    "Do not output a final passage ranking.",
                    "Do not include direct passage-id ordering hints such as '[3] should be first'.",
                    "The new strategy must be general instructions for the student, not an answer to this specific instance.",
                    f"The new strategy must preserve the student's required output format: {required_output_format}.",
                    "Candidate status means: current = the strategy used for the latest failed rollout; chosen = a strategy tried in an earlier repair round; candidate = a library strategy not yet used in this repair trace.",
                    "Use select_candidate_strategy only for an untried candidate because current and chosen strategies have already failed when used unchanged.",
                    "For select_candidate_strategy, output selected_candidate_key and set new_strategy to null; the system will use that candidate card unchanged.",
                    "For rewrite_library_strategy, selected_candidate_key may refer to current, chosen, or candidate, but new_strategy must be a complete revised card and must differ from all existing library cards.",
                    "For create_new_strategy, selected_candidate_key must be null and new_strategy must be a complete new card not already in the library.",
                ],
            }
        ),
        "ranking_problem": request.get("ranking_problem", {}),
        "current_strategy": request.get("current_strategy", {}),
        "candidate_strategy_library": request.get("candidate_strategy_library", []),
        "student_best_rollout_under_current_strategy": request.get("student_best_rollout_under_current_strategy", {}),
        "output_schema": {
            "diagnosis_type": "strategy_description_bad | wrong_strategy_selected | strategy_missing",
            "action": "select_candidate_strategy | rewrite_library_strategy | create_new_strategy",
            "selected_candidate_key": "string for select/rewrite, null for create",
            "failure_analysis": "short diagnostic explanation",
            "new_strategy": (
                "null for select_candidate_strategy; otherwise a complete strategy object with keys: "
                "strategy_name, objective, when_to_apply, query_signals, retention_rule, evidence_to_promote, "
                "evidence_to_demote, ranking_procedure, tie_breaking_rule, fallback_behavior, avoid_when"
            ),
            "confidence": 0.0,
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def build_repair_llm_payload(request: dict[str, Any], config: Any) -> dict[str, Any]:
    payload = {
        "model": _cfg_get(config, "model"),
        "messages": _build_messages(request),
        "temperature": float(_cfg_get(config, "temperature", 0.2)),
        "max_tokens": int(_cfg_get(config, "max_tokens", 2048)),
    }
    if bool(_cfg_get(config, "use_response_format", False)):
        payload["response_format"] = {"type": "json_object"}
    enable_thinking = _cfg_get(config, "enable_thinking", None)
    if enable_thinking is not None:
        payload["enable_thinking"] = bool(enable_thinking)
    return payload


class RepairLLMClient:
    def __init__(self, config: Any):
        self.config = config or {}
        self.base_url = _cfg_get(config, "base_url")
        self.model = _cfg_get(config, "model")
        api_key_env = _cfg_get(config, "api_key_env", "GUIDED_REPAIR_LLM_API_KEY")
        self.api_key = os.environ.get(api_key_env or "")
        self.timeout_seconds = float(_cfg_get(config, "timeout_seconds", 60))
        self.max_retries = int(_cfg_get(config, "max_retries", 3))
        self.retry_backoff_seconds = float(_cfg_get(config, "retry_backoff_seconds", 2))
        self.concurrency = max(1, int(_cfg_get(config, "concurrency", 8)))
        self.temperature = float(_cfg_get(config, "temperature", 0.2))
        self.max_tokens = int(_cfg_get(config, "max_tokens", 2048))
        self.use_response_format = bool(_cfg_get(config, "use_response_format", False))
        self.enable_thinking = _cfg_get(config, "enable_thinking", None)

        if not self.model:
            raise ValueError("guided repair llm model is required.")
        if not self.api_key:
            raise ValueError(f"guided repair llm api key env {api_key_env!r} is not set.")

        self.url = _chat_completions_url(self.base_url)

    async def repair_batch(self, requests: list[dict[str, Any]]) -> list[RepairLLMResult]:
        semaphore = asyncio.Semaphore(self.concurrency)
        print(
            f"[guided_repair_llm] batch_start request_count={len(requests)} "
            f"concurrency={self.concurrency} timeout_seconds={self.timeout_seconds} max_retries={self.max_retries}",
            flush=True,
        )
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            tasks = [self._repair_one(client, semaphore, request_idx, request) for request_idx, request in enumerate(requests)]
            return await asyncio.gather(*tasks)

    async def _repair_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        request_idx: int,
        request: dict[str, Any],
    ) -> RepairLLMResult:
        start_time = time.monotonic()
        retry_count = 0
        last_error_type = "unknown"
        last_error_message = ""
        attempts = []
        last_raw_response = None

        async with semaphore:
            ranking_problem = request.get("ranking_problem") or {}
            best_rollout = request.get("student_best_rollout_under_current_strategy") or {}
            print(
                f"[guided_repair_llm] request_start idx={request_idx} "
                f"problem_chars={len(str(ranking_problem.get('base_user_instruction', '')))} "
                f"rollout_chars={len(str(best_rollout.get('response', '')))} "
                f"candidate_count={len(request.get('candidate_strategy_library') or [])}",
                flush=True,
            )
            for attempt in range(self.max_retries + 1):
                retry_count = attempt
                attempt_raw_response = None
                try:
                    print(f"[guided_repair_llm] attempt_start idx={request_idx} attempt={attempt}", flush=True)
                    content = await self._send_request(client, request)
                    attempt_raw_response = content
                    last_raw_response = content
                    decision = _validate_decision(_extract_json_object(content), request)
                    attempts.append(
                        {
                            "attempt": attempt,
                            "success": True,
                            "raw_response": attempt_raw_response,
                        }
                    )
                    print(
                        f"[guided_repair_llm] request_success idx={request_idx} attempt={attempt} "
                        f"latency_s={time.monotonic() - start_time:.3f} "
                        f"action={decision.get('action')} diagnosis_type={decision.get('diagnosis_type')} "
                        f"raw_chars={len(content)}",
                        flush=True,
                    )
                    return RepairLLMResult(
                        success=True,
                        decision=decision,
                        raw_response=content,
                        latency_s=time.monotonic() - start_time,
                        retry_count=retry_count,
                        attempts=attempts,
                    )
                except (httpx.TimeoutException,) as exc:
                    last_error_type = "timeout"
                    last_error_message = str(exc)
                except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                    last_error_type = "connection_error"
                    last_error_message = str(exc)
                except httpx.HTTPStatusError as exc:
                    last_error_type = f"http_{exc.response.status_code}"
                    last_error_message = exc.response.text[:500]
                    if exc.response.status_code not in {429, 500, 502, 503, 504}:
                        break
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error_type = "invalid_json_or_schema"
                    last_error_message = str(exc)
                except Exception as exc:
                    last_error_type = type(exc).__name__
                    last_error_message = str(exc)

                print(
                    f"[guided_repair_llm] attempt_failed idx={request_idx} attempt={attempt} "
                    f"error_type={last_error_type} error_message={last_error_message[:300]}",
                    flush=True,
                )
                attempts.append(
                    {
                        "attempt": attempt,
                        "success": False,
                        "error_type": last_error_type,
                        "error_message": last_error_message,
                        "raw_response": attempt_raw_response,
                    }
                )

                if attempt < self.max_retries:
                    backoff = self.retry_backoff_seconds * (2**attempt)
                    backoff += random.uniform(0.0, self.retry_backoff_seconds)
                    print(
                        f"[guided_repair_llm] retry_sleep idx={request_idx} attempt={attempt} backoff_s={backoff:.3f}",
                        flush=True,
                    )
                    await asyncio.sleep(backoff)

        print(
            f"[guided_repair_llm] request_failed idx={request_idx} retry_count={retry_count} "
            f"latency_s={time.monotonic() - start_time:.3f} error_type={last_error_type} "
            f"error_message={last_error_message[:300]}",
            flush=True,
        )
        return RepairLLMResult(
            success=False,
            raw_response=last_raw_response,
            error_type=last_error_type,
            error_message=last_error_message,
            latency_s=time.monotonic() - start_time,
            retry_count=retry_count,
            attempts=attempts,
        )

    async def _send_request(self, client: httpx.AsyncClient, request: dict[str, Any]) -> str:
        payload = build_repair_llm_payload(request, self.config)

        response = await client.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("LLM response has no choices.")

        choice = choices[0]
        if "message" in choice and isinstance(choice["message"], dict):
            return choice["message"].get("content") or ""
        if "text" in choice:
            return choice.get("text") or ""
        raise ValueError("LLM response choice has no message.content or text.")


def run_repair_batch(requests: list[dict[str, Any]], config: Any) -> list[RepairLLMResult]:
    if not requests:
        return []
    return asyncio.run(RepairLLMClient(config).repair_batch(requests))
