"""
Sard reward for partial listwise reranking.

Expected output:
<thinking>...</thinking>
<answer>[1] > [2] > ...</answer>

The model is allowed to return a partial ranking containing only relevant
passages. NDCG@k and Recall@k use the top-k prefix of that partial ranking.

Ground-truth schema:
- graded_relevance: per-passage graded relevance gains used by NDCG and Recall.
"""

import math
import re
from typing import Iterable, List, Sequence


def _extract_thinking(text: str) -> str:
    match = re.search(r"<thinking>(.*?)</thinking>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_raw_ranking_indices(answer: str) -> List[int]:
    bracketed = [int(x) for x in re.findall(r"\[(\d+)\]", answer or "")]
    if bracketed:
        return bracketed
    return [int(x) for x in re.findall(r"\b\d+\b", answer or "")]


def _filter_valid_indices(indices: Iterable[int], num_docs: int) -> List[int]:
    seen = set()
    valid = []
    for idx in indices:
        if idx < 1 or idx > num_docs or idx in seen:
            continue
        seen.add(idx)
        valid.append(idx)
    return valid


def _dcg(gains: Sequence[float]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))


def _ndcg_at_k(pred_indices: Sequence[int], graded_relevance: Sequence[float], k: int) -> float:
    if not graded_relevance or not pred_indices:
        return 0.0

    pred_top = pred_indices[:k]
    pred_gains = [graded_relevance[idx - 1] for idx in pred_top]
    dcg = _dcg(pred_gains)

    ideal_gains = sorted(graded_relevance, reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(pred_indices: Sequence[int], graded_relevance: Sequence[float], k: int) -> float:
    relevant = {idx + 1 for idx, rel in enumerate(graded_relevance) if rel > 0}
    if not relevant or not pred_indices:
        return 0.0

    pred_top = set(pred_indices[:k])
    denom = min(k, len(relevant))
    return len(pred_top & relevant) / denom if denom > 0 else 0.0


def _get_graded_relevance(ground_truth):
    if not isinstance(ground_truth, dict):
        raise ValueError("sard_reward expects ground_truth to be a dict containing graded_relevance.")

    graded_relevance = ground_truth.get("graded_relevance")
    if hasattr(graded_relevance, "tolist"):
        graded_relevance = graded_relevance.tolist()

    if not isinstance(graded_relevance, (list, tuple)):
        raise ValueError("sard_reward expects ground_truth.graded_relevance to be a list.")

    graded_relevance = [float(rel) for rel in graded_relevance]
    for rel in graded_relevance:
        if not math.isfinite(rel) or rel < 0:
            raise ValueError(f"sard_reward graded_relevance values must be finite and non-negative, got {rel}.")
    return graded_relevance


def compute_score(solution_str: str, ground_truth, extra_info=None):
    graded_relevance = _get_graded_relevance(ground_truth)
    num_docs = len(graded_relevance)

    k = 10
    ndcg_weight = 1.0
    recall_weight = 0.2
    if isinstance(ground_truth, dict):
        k = int(ground_truth.get("k", k))
        ndcg_weight = float(ground_truth.get("ndcg_weight", ndcg_weight))
        recall_weight = float(ground_truth.get("recall_weight", recall_weight))
    if k <= 0:
        raise ValueError("sard_reward requires k > 0.")

    thinking = _extract_thinking(solution_str)
    answer = _extract_answer(solution_str)
    has_thinking = bool(thinking)
    has_answer = bool(answer)

    raw_indices = _extract_raw_ranking_indices(answer) if has_answer else []
    raw_parse_ok = bool(raw_indices)
    has_format_error = (not has_thinking) or (not has_answer) or (not raw_parse_ok)

    pred_indices = _filter_valid_indices(raw_indices, num_docs)
    valid_parse_ok = bool(pred_indices)

    has_logic_error = False
    if raw_parse_ok:
        has_oob = any(idx < 1 or idx > num_docs for idx in raw_indices)
        has_duplicate = len(raw_indices) != len(set(raw_indices))
        has_logic_error = has_oob or has_duplicate

    ndcg = _ndcg_at_k(pred_indices, graded_relevance, k) if valid_parse_ok else 0.0
    recall = _recall_at_k(pred_indices, graded_relevance, k) if valid_parse_ok else 0.0

    if has_format_error:
        score = -1.0
    elif has_logic_error:
        score = -0.5
    else:
        score = ndcg_weight * ndcg + recall_weight * recall

    return {
        "score": score,
        f"ndcg@{k}": ndcg,
        f"recall@{k}": recall,
        "parse_ok": float(raw_parse_ok),
        "valid_parse_ok": float(valid_parse_ok),
        "num_raw_pred": len(raw_indices),
        "num_pred": len(pred_indices),
        "format_ok": float(not has_format_error),
        "logic_ok": float(not has_logic_error),
        "has_thinking": float(has_thinking),
        "has_answer": float(has_answer),
        "has_oob": float(any(idx < 1 or idx > num_docs for idx in raw_indices)) if raw_parse_ok else 0.0,
        "has_duplicate": float(len(raw_indices) != len(set(raw_indices))) if raw_parse_ok else 0.0,
        "k_effective": min(k, len(pred_indices)) if valid_parse_ok else 0,
    }
