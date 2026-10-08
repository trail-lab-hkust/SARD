# SARD

## Dependencies

SARD is implemented on top of two open-source training frameworks:

- **LLaMA-Factory**: used for the SFT warm-up stage. The relevant entry script
  and config are in `sft/`.
- **verl**: used for GRPO-based RL training. The SARD-specific changes are
  provided as an overlay under `rl/verl_patch/`.

The files in `rl/verl_patch/` are the modified `verl` files needed to inspect
the routing, fallback, guidance, repair, off-policy loss, and reward logic.

## Structure

- `sft/`: SFT script and LLaMA-Factory config.
- `rl/scripts/`: GRPO, fallback, guidance, and repair training entries.
- `rl/verl_patch/`: selected `verl` files modified for SARD.
- `examples/`: five real SFT examples and five real RL examples.

## Main RL Setting

`rl/scripts/train_repair.sh` is the main SARD setting:

```text
repair.max_rounds=3
repair.llm.model=deepseek-v3.2-exp
train_auxiliary_groups=True
auxiliary_group_loss_weight=1.0
teacher_fallback_aux_group=none
```

## Reward

The reward expects:

```text
<thinking>...</thinking>
<answer>[1] > [3] > ...</answer>
```

It uses `reward_model.ground_truth.graded_relevance` and computes:

```text
score = NDCG@k + 0.2 * Recall@k
```

Format errors receive `-1.0`; duplicate or out-of-range passage ids receive
`-0.5`.
