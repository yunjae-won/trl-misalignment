# TRL Misalignment

Small overlay package for monitoring vocab-level misalignment during TRL
policy training.

The implementation is intentionally self-contained:

- `trl_misalignment.metrics` contains the provided
  `batched_vocablevel_misalignment` function and a padded completion helper
  that aggregates token rows back to prompt-level sums.
- `trl_misalignment.rewards` computes vocab rewards
  `R = log pi_w - log pi_l` through two `logprob-engine` backends.
- `trl_misalignment.trainers` provides trainer subclasses for GRPO and
  Online-DPO. GRPO supports optional `J` backpropagation; Online-DPO currently
  supports monitoring-only.

## Reward servers

`logprob-engine` already has the vocab-level path in its `LogprobEngine`; this
repo adds a tiny server entrypoint that exposes it over HTTP:

```bash
CUDA_VISIBLE_DEVICES=6 python -m trl_misalignment.serve_vocab_logprobs \
  --model /path/to/pi_w --device cuda:0 --port 8101

CUDA_VISIBLE_DEVICES=7 python -m trl_misalignment.serve_vocab_logprobs \
  --model /path/to/pi_l --device cuda:0 --port 8102
```

Then launch training on the policy GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
  examples/token_vocab_reward_training.py \
  --trainer grpo \
  --model /path/to/4b-policy \
  --winner-url http://127.0.0.1:8101 \
  --loser-url http://127.0.0.1:8102 \
  --use-vllm \
  --vllm-mode server \
  --vllm-server-port 8000
```

For colocated vLLM, use TRL's normal `--vllm-mode colocate` settings and keep
the reward servers on separate GPUs.

## Autograd behavior

`VocabMisalignmentConfig(backprop_j=False)` is the default monitor mode: policy
logits, reference logits, reward tensors, and gamma are detached. With GRPO,
`backprop_j=True` keeps the policy side of `J` attached and adds the per-prompt
`J` mean to the GRPO loss. The gamma solve remains under `torch.no_grad()`.
