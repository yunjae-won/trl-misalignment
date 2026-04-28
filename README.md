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
  Online-DPO. GRPO and Online-DPO support optional `J` backpropagation with a
  configurable coefficient.
- `docs/research_process.md` records the active experimental protocol, metrics,
  storage layout, and follow-up ablation rules.

## Full ablation script

The recommended paper-run entrypoint is:

```bash
scripts/run_misalignment_ablation.sh
```

Defaults:

- seed: `20260428`
- policy model: `yunjae-won/ubq30i_qwen4b_sft_both`
- reward models: `yunjae-won/ubq30i_qwen4b_sft_yw` and `yunjae-won/ubq30i_qwen4b_sft_yl`
- dataset: `trl-lib/ultrafeedback-prompt`
- wandb project: `trl-misalignment-grpo-online-dpo-20260428`
- GPUs: policy training on `0,1,2,3`, vLLM on `4,5`, reward servers on `6` and `7`
- checkpoints: local `runs/misalignment_ablation/<run_id>`
- persistent metadata/logs/summaries: `/yj_data/trl_misalignment/<run_id>`
- tokenization audit samples: `/yj_data/trl_misalignment/<run_id>/tokenization_debug/*.jsonl`

The script runs vanilla GRPO/Online-DPO with `J` detached and auxiliary-loss
ablations with `--misalignment-loss-coef`. Override run scope with environment
variables such as `MAX_STEPS`, `MAX_TRAIN_SAMPLES`, `GRPO_COEFS`,
`ONLINE_DPO_COEFS`, `DEBUG_TOKENIZATION_SAMPLES`, or `ALGOS`.

For each run group, it writes `summary_metrics.csv`, `analysis.csv`, and
`analysis.md` under `/yj_data/trl_misalignment/<run_id>`.

## Reward servers

The vendored `logprob_engine` path is optimized for full-vocab rewards: compiled
forwards return dense tensors, HTTP binary responses stay as NumPy arrays, and
`npz` is intentionally uncompressed. The JSON/list path is still available for
debugging, but training should use the default `npz` format.

```bash
CUDA_VISIBLE_DEVICES=6 python -m trl_misalignment.serve_vocab_logprobs \
  --model /path/to/pi_w --device cuda:0 --port 8101 \
  --dtype bfloat16 --logprob-dtype float32

CUDA_VISIBLE_DEVICES=7 python -m trl_misalignment.serve_vocab_logprobs \
  --model /path/to/pi_l --device cuda:0 --port 8102 \
  --dtype bfloat16 --logprob-dtype float32
```

`scripts/run_misalignment_ablation.sh` starts both reward servers with
`torch.compile` enabled by default and sends a small warmup request before
policy training, so the slow first compile does not land inside the first
training step. Set `REWARD_COMPILE=0` or `REWARD_WARMUP=0` only for debugging.

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
`backprop_j=True` keeps the policy side of `J` attached and adds
`j_loss_coef * mean(J)` to the GRPO loss. For Online-DPO, the trainer copies
the upstream training step so the same auxiliary term can be added before
`accelerator.backward`. The gamma solve remains under `torch.no_grad()`.
