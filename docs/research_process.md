# Misalignment Research Process

This repository is being used to study whether GRPO and onlineDPO amplify
vocab-level reward misalignment, and whether adding the misalignment term `J`
to the training loss reduces that behavior.

## Fixed Experimental Setup

- Date/seed: `2026-04-28`, seed `20260428`
- Dataset: `trl-lib/ultrafeedback-prompt`
- Policy initialization: `yunjae-won/ubq30i_qwen4b_sft_both`
- Winner reward model: `yunjae-won/ubq30i_qwen4b_sft_yw`
- Loser reward model: `yunjae-won/ubq30i_qwen4b_sft_yl`
- Policy training GPUs: `0,1,2,3`
- vLLM generation GPUs: `4,5`
- Reward model GPUs: `6,7`
- wandb project: `trl-misalignment-grpo-online-dpo-20260428`
- Persistent metadata/logs: `/yj_data/trl_misalignment/<run_id>`
- Local checkpoints only: `/root/trl-misalignment/runs/misalignment_ablation/<run_id>`

The Qwen chat template can leave a newline token after the assistant EOS when
used directly for a prompt/completion pair. Training therefore normalizes all
reward-model completion IDs by removing non-EOS pad tokens and truncating after
the last completion-side EOS before sending them to the vocab logprob reward
servers. This normalization is applied only to the completion IDs, never to the
prompt prefix, because the prompt itself may contain EOS tokens.

## Metrics Logged During Training

Primary objective metrics:

- GRPO scalar reward: `reward`, `rewards/token_vocab_reward/*`
- onlineDPO scalar reward: `objective/scores`, `objective/scores_margin`
- policy optimization health: `loss`, `grad_norm`, trainer KL metrics

Vocab-level misalignment metrics:

- `misalignment/J`: prompt-summed J
- `misalignment/J_per_token`: length-normalized J, preferred for comparing runs
- `misalignment/reward_a*`, `misalignment/reward_b*`: expected vocab reward under
  policy/reference token distributions
- `misalignment/reward_improvement*`: policy expected reward minus reference
  expected reward
- `misalignment/*_kl*`, `misalignment/js_divergence*`,
  `misalignment/tv_distance*`: policy/reference distribution shift
- `misalignment/gamma_bracketed_rate`: numerical health of the gamma solve
- `misalignment/reward_vocab_std_per_token`,
  `misalignment/reward_vocab_abs_max_per_token`: reward-model sharpness/spread
- `misalignment/policy_reference_top1_agreement_rate`,
  `misalignment/policy_top_is_reward_top_rate`: whether the policy is moving
  toward high-reward top tokens rather than just increasing sampled-token reward

Auxiliary-loss runs also log:

- `misalignment/J_aux_loss_raw`
- `misalignment/J_aux_loss`

Tokenization/debug metrics:

- `tokenization/prompt_length`
- `tokenization/completion_length`
- `tokenization/normalized_completion_length`
- `tokenization/prompt_contains_eos_rate`
- `tokenization/completion_has_eos_rate`
- `tokenization/completion_ends_eos_rate`
- `tokenization/trailing_tokens_after_last_completion_eos`

Reward-engine timing metrics:

- `reward_engine/wall_seconds`
- `reward_engine/scoring_wall_seconds`
- `reward_engine/scoring_reward_seconds`
- `reward_engine/winner_seconds`
- `reward_engine/loser_seconds`
- `reward_engine/score_and_cache_seconds`
- `reward_engine/missing_items`, `reward_engine/cached_items`
- `reward_engine/missing_tokens`, `reward_engine/cached_tokens`

`scoring_*` is the preferred speed metric when the trainer reuses a cached
reward tensor for the J monitor. GRPO first calls the scalar reward function and
then the monitor; the monitor call is often a cache hit, so plain
`reward_engine/wall_seconds` can measure cache lookup time instead of the actual
reward-model scoring request.

Each run also writes a small JSONL tokenization audit under
`/yj_data/trl_misalignment/<run_id>/tokenization_debug/<run_name>.jsonl`. It
contains sampled dataset prompt tokenization plus the first rollout's raw and
normalized completion IDs/text.

## Run Plan

The base ablation is:

| algorithm | J coefficient |
| --- | --- |
| GRPO | `0`, `0.001`, `0.003` |
| onlineDPO | `0`, `0.001`, `0.003` |

The coefficient `0` condition monitors `J` only and detaches the metric path
from training. Nonzero coefficients add `coef * mean(J)` to the policy loss.

The first full research pass should be treated as a pilot ablation. After it
finishes, compare nonzero coefficients against each algorithm's coefficient-0
baseline using:

- lower `misalignment/J_per_token`
- unchanged or improved scalar reward/scores
- stable `grad_norm`
- `gamma_bracketed_rate` close to `1`

Follow-up ablations should narrow around the smallest coefficient that reduces
`J_per_token` without hurting scalar reward. If both `0.001` and `0.003` reduce
J, add a higher coefficient such as `0.01`. If both hurt reward, add smaller
coefficients such as `0.0001` and `0.0003`.

## Execution Log

### 2026-04-28 Vocab Reward Engine Fix

Purpose: remove Python/CPU serialization from the full-vocab reward path and
make `torch.compile` useful for the reward servers.

Changes:

- Moved per-item `split` out of the compiled `LogprobEngine.forward`; compiled
  vocab/token forwards now return one dense tensor.
- Added `process_tensors` and `process_arrays` so training does not inflate
  `[tokens, vocab]` logprobs into Python lists.
- Changed HTTP `format=npz` to uncompressed NPZ and kept `npz_compressed` as an
  opt-in debug format.
- Made the reward provider call winner and loser servers in parallel, while each
  server serializes calls to its own model with a process-local lock.
- Enabled reward-server compile in the ablation script and added a warmup request
  before policy training.

Validation:

- `python -m unittest tests.test_rewards tests.test_logprob_engine`: 7 tests OK.
- Synthetic serialization payload, 4 arrays of shape `[64, 8192]`:
  uncompressed NPZ `0.0135s`, compressed NPZ `0.3497s`, Python `tolist`
  `0.0545s`.
- Real reward-model smoke test on GPU 6 with
  `yunjae-won/ubq30i_qwen4b_sft_yw`: first compiled request `121.197s`, second
  warmed request `0.028s`, output shape `[3, 151936]`, dtype `float32`,
  finite values, row logsumexp `0.000000`.

### 2026-04-28 Optimized Engine Speed Check

Purpose: resume experiments against the standalone optimized
`logprob-engine==0.1.1` pushed to `https://github.com/yunjae-won/logprob-engine`.

Changes before resuming:

- Switched this repo back to the GitHub `logprob-engine` dependency instead of
  the temporary vendored copy.
- Confirmed imports resolve to `/root/logprob-engine/logprob_engine/__init__.py`
  with `process_arrays` and `LogprobClient.logprob_arrays`.
- Updated reward warmup to compile a shape closer to trainer traffic:
  `REWARD_WARMUP_ITEMS=4` and `REWARD_WARMUP_TOKENS=MAX_COMPLETION_LENGTH` by
  default.

Matched GRPO baseline speed check:

```bash
RUN_ID=optimized-logprob-speedcheck-seed20260428 \
ALGOS=grpo GRPO_COEFS=0 \
MAX_STEPS=10 MAX_TRAIN_SAMPLES=160 MAX_PROMPT_LENGTH=512 \
MAX_COMPLETION_LENGTH=64 SAVE_STEPS=10 LOGGING_STEPS=1 \
DEBUG_TOKENIZATION_SAMPLES=4 REWARD_COMPILE=1 REWARD_WARMUP=1 \
scripts/run_misalignment_ablation.sh
```

Results:

- Pre-fix matched pilot (`pilot-ablation-v1-seed20260428`): mean step time
  `76.887s`, steady-state mean excluding step 1 `76.697s`, train runtime
  `783.596s`, train steps/sec `0.013`.
- Optimized engine run (`optimized-logprob-speedcheck-seed20260428`): mean step
  time `18.773s`, steady-state mean excluding step 1 `5.033s`, train runtime
  `202.860s`, train steps/sec `0.049`.
- Speedup: `4.10x` including the first in-loop warmup step; `15.24x` on
  steady-state steps.
- Reward warmup itself took about `58s` per reward server for the earlier
  1-item, 8-token warmup. Step 1 still took `142.434s`, indicating a
  shape-specific compile; subsequent steps were `4.865s` to `5.811s`.

### 2026-04-28 Optimized Pilot Ablation v2

Purpose: resume the GRPO/onlineDPO misalignment ablation using the standalone
optimized `logprob-engine`, with shape-matched reward warmup enabled.

Command:

```bash
RUN_ID=optimized-pilot-ablation-v2-seed20260428 \
ALGOS='grpo online_dpo' \
GRPO_COEFS='0 0.001 0.003' \
ONLINE_DPO_COEFS='0 0.001 0.003' \
MAX_STEPS=10 MAX_TRAIN_SAMPLES=160 MAX_PROMPT_LENGTH=512 \
MAX_COMPLETION_LENGTH=64 SAVE_STEPS=10 LOGGING_STEPS=1 \
DEBUG_TOKENIZATION_SAMPLES=4 REWARD_COMPILE=1 REWARD_WARMUP=1 \
scripts/run_misalignment_ablation.sh
```

Artifacts:

- Persistent metadata: `/yj_data/trl_misalignment/optimized-pilot-ablation-v2-seed20260428`
- Local checkpoints: `/root/trl-misalignment/runs/misalignment_ablation/optimized-pilot-ablation-v2-seed20260428`
- Additional speed table: `/yj_data/trl_misalignment/optimized-pilot-ablation-v2-seed20260428/speed_metrics.csv`

Speed findings:

| run | runtime | steps/sec | reward wall/step | final J/token |
| --- | ---: | ---: | ---: | ---: |
| GRPO `0` | `68.659s` | `0.146` | `0.0903s` | `7.128e-04` |
| GRPO `0.001` | `87.343s` | `0.114` | `0.0329s` | `8.272e-04` |
| GRPO `0.003` | `82.665s` | `0.121` | `0.0353s` | `8.031e-04` |
| onlineDPO `0` | `98.574s` | `0.101` | `0.0451s` | `8.634e-04` |
| onlineDPO `0.001` | `97.996s` | `0.102` | `0.0478s` | `8.396e-04` |
| onlineDPO `0.003` | `97.352s` | `0.103` | `0.0427s` | `7.796e-04` |

Relative to the pre-fix matched GRPO baseline (`783.596s` for 10 steps), the
current warmed GRPO baseline is `11.41x` faster by train runtime. Relative to
the first optimized speed check (`202.860s`), shape-matched warmup improves the
10-step runtime another `2.95x` and removes the in-loop compile spike.

Tokenization audit:

- All sampled prompts contain the expected prompt-side chat-template EOS token.
- No sampled 64-token debug rollout contained a completion-side EOS because the
  sampled generations mostly hit the max completion cap; this is expected for
  the short pilot.
- `trailing_tokens_after_last_completion_eos` was `0` for every logged training
  step and every JSONL audit sample, so completion normalization did not leave
  post-EOS newline/pad tokens in reward-model inputs.
- The audit confirms prompt EOS tokens are kept in the prompt prefix and never
  confused with completion EOS handling.

Interpretation:

- The optimized vocab reward computation is no longer the bottleneck. For the
  main pilot runs, the reward provider accounts for roughly `0.03s` to `0.05s`
  per trainer step, while total trainer steps are around `6s` to `10s`.
- The 10-step GRPO coefficient comparison is too noisy for a behavioral claim:
  final `J/token` is higher for `0.001` and `0.003` than for the coefficient-0
  baseline, while mean `J/token` across steps is nearly tied.
- The onlineDPO coefficient comparison is directionally more promising at the
  final step (`0.003` has the lowest final `J/token`), but the mean over the
  short pilot is still noisy. A longer intermediate run is needed before making
  paper-facing claims.

### 2026-04-28 Debug Tokenization Pass

Purpose: verify the prompt/completion boundary, EOS handling, reward-provider
normalization, wandb logging, and service cleanup before launching longer
ablations.

Commands:

```bash
RUN_ID=debug-tokenization-seed20260428 \
ALGOS='grpo online_dpo' GRPO_COEFS=0 ONLINE_DPO_COEFS=0 \
MAX_STEPS=1 MAX_TRAIN_SAMPLES=4 MAX_PROMPT_LENGTH=256 \
MAX_COMPLETION_LENGTH=32 SAVE_STEPS=1 LOGGING_STEPS=1 \
DEBUG_TOKENIZATION_SAMPLES=4 scripts/run_misalignment_ablation.sh

RUN_ID=debug-tokenization-online-dpo-seed20260428 \
ALGOS=online_dpo ONLINE_DPO_COEFS=0 \
MAX_STEPS=1 MAX_TRAIN_SAMPLES=4 MAX_PROMPT_LENGTH=256 \
MAX_COMPLETION_LENGTH=32 SAVE_STEPS=1 LOGGING_STEPS=1 \
DEBUG_TOKENIZATION_SAMPLES=4 scripts/run_misalignment_ablation.sh
```

Findings:

- Dataset prompt tokenization correctly includes chat-template EOS tokens in
  the prompt prefix; these are expected and are independent of completion-side
  normalization.
- GRPO with 32-token completions produced no completion EOS in the sampled
  rollout, so the reward provider left completions untrimmed.
- onlineDPO with 32-token completions produced EOS in part of the sampled
  rollout; normalized completion IDs ended exactly at the completion EOS.
- `gamma_bracketed_rate` was `1.0` in both debug runs.
- The first mixed debug run exposed a vLLM cleanup issue: worker children from
  the GRPO vLLM process could remain alive and hold GPUs 4-5. The launcher now
  starts services in separate process groups, kills the whole group between
  runs, and fails fast on vLLM startup tracebacks.

Debug artifacts:

- GRPO: `/yj_data/trl_misalignment/debug-tokenization-seed20260428`
- onlineDPO: `/yj_data/trl_misalignment/debug-tokenization-online-dpo-seed20260428`

### 2026-04-28 Optimized Midrun Ablation v3

Purpose: run a longer 50-step ablation after the optimized standalone
`logprob-engine` was pushed, using the pilot result to test stronger
coefficients (`0.003`, `0.01`) against detached-monitor baselines.

Command:

```bash
RUN_ID=optimized-midrun-ablation-v3-seed20260428 \
ALGOS='grpo online_dpo' \
GRPO_COEFS='0 0.003 0.01' \
ONLINE_DPO_COEFS='0 0.003 0.01' \
MAX_STEPS=50 MAX_TRAIN_SAMPLES=800 MAX_PROMPT_LENGTH=512 \
MAX_COMPLETION_LENGTH=64 SAVE_STEPS=50 LOGGING_STEPS=5 \
DEBUG_TOKENIZATION_SAMPLES=4 REWARD_COMPILE=1 REWARD_WARMUP=1 \
scripts/run_misalignment_ablation.sh
```

Artifacts:

- Persistent metadata: `/yj_data/trl_misalignment/optimized-midrun-ablation-v3-seed20260428`
- Local checkpoints: `/root/trl-misalignment/runs/misalignment_ablation/optimized-midrun-ablation-v3-seed20260428`
- Speed table: `/yj_data/trl_misalignment/optimized-midrun-ablation-v3-seed20260428/speed_metrics.csv`
- Tokenization audit: `/yj_data/trl_misalignment/optimized-midrun-ablation-v3-seed20260428/tokenization_audit.csv`

Final metric summary:

| algorithm | coef | reward/scores | final J/token | delta vs baseline | mean J/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO | `0` | `-0.932` | `9.321e-04` |  | `9.008e-04` |
| GRPO | `0.003` | `-2.478` | `9.213e-04` | `-1.078e-05` | `8.839e-04` |
| GRPO | `0.01` | `-2.340` | `9.296e-04` | `-2.533e-06` | `8.809e-04` |
| onlineDPO | `0` | `0.269` | `9.180e-04` |  | `8.809e-04` |
| onlineDPO | `0.003` | `0.554` | `9.668e-04` | `+4.878e-05` | `9.042e-04` |
| onlineDPO | `0.01` | `-0.571` | `9.888e-04` | `+7.081e-05` | `9.053e-04` |

Speed summary:

| run | runtime | steps/sec | reward wall/logged step |
| --- | ---: | ---: | ---: |
| GRPO `0` | `269.948s` | `0.185` | `0.0937s` |
| GRPO `0.003` | `372.880s` | `0.134` | `0.0355s` |
| GRPO `0.01` | `354.843s` | `0.141` | `0.0373s` |
| onlineDPO `0` | `429.457s` | `0.116` | `0.0461s` |
| onlineDPO `0.003` | `428.546s` | `0.117` | `0.0454s` |
| onlineDPO `0.01` | `425.110s` | `0.118` | `0.0437s` |

Relative to the pre-fix matched GRPO pilot (`0.013` steps/sec), the optimized
50-step GRPO baseline reached `0.185` steps/sec, about `14.2x` faster by
trainer throughput. Reward scoring is now small compared with policy
optimization and generation: `0.04s` to `0.09s` per logged step, while full
steps are roughly `5s` for detached GRPO, `6.8s` to `7.1s` for GRPO with the
auxiliary gradient path, and `8.5s` for onlineDPO.

Tokenization audit:

- All sampled dataset prompts contained the expected prompt-side EOS token with
  `prompt_eos_count=1`.
- Completion-side EOS was rare in the sampled debug rollouts at
  `max_completion_length=64`, but the training metrics saw nonzero completion
  EOS rates for onlineDPO.
- `tokenization/trailing_tokens_after_last_completion_eos` was `0` in every
  logged step and every JSONL audit row.
- Prompt-side EOS tokens remained in the prompt prefix and were not treated as
  completion EOS.

Interpretation:

- The optimized logprob engine is no longer the bottleneck; remaining runtime
  is dominated by vLLM generation, policy forward/backward, and TRL trainer
  mechanics.
- GRPO `0.003` and `0.01` give tiny final-step J/token reductions, but the
  effect is far smaller than run-to-run noise and comes with worse scalar
  reward. This is not a useful paper-facing improvement yet.
- onlineDPO `0.003` improves scalar score on this seed but increases J/token;
  `0.01` increases J/token further and hurts scalar score. The simple
  `coef * J` auxiliary loss does not improve onlineDPO misalignment in this
  50-step setting.
- Next experiment should measure variance rather than increase the coefficient:
  run a second seed for the coefficient-0 baselines and the smallest nonzero
  candidates, or add a held-out evaluation pass that scores the same prompts
  from each checkpoint before making algorithmic claims.

### 2026-04-28 Long-Sequence Setup

Purpose: move from short 64-token pilots to paper-relevant long generations:
200 optimizer steps, 1024 max generated tokens, max model length capped at
4096, logging every step, and the requested optimizer schedule
(`1e-6`, `constant_with_warmup`, 10% warmup, grad clip `10.0`).

Implementation changes:

- Added launcher/training args for `lr_scheduler_type`, `max_grad_norm`,
  `max_model_length`, gradient checkpointing, FSDP, and DeepSpeed config paths.
- Added a paired reward server, `python -m trl_misalignment.serve_vocab_reward`,
  which loads winner and loser reward models and returns the vocab reward
  difference directly. This keeps the vocab-level reward model semantics but
  avoids sending both full-vocab tensors over HTTP and caching three copies.
- Changed the reward provider cache to retain only `winner - loser`; the cached
  winner/loser tensors were not used by the trainers and were too expensive for
  1024-token completions.
- Added `reward_engine/scoring_*` timing metrics so cached monitor calls still
  expose the actual reward scoring cost from the scalar reward call.
- The launcher can still run the old two-server mode with
  `REWARD_SERVER_MODE=separate`; the long run uses `paired`.

Model-parallelism check:

```bash
RUN_ID=smoke-longseq-fsdp-paired-v1-seed20260428 \
ALGOS=grpo GRPO_COEFS=0 MAX_STEPS=1 \
MAX_PROMPT_LENGTH=1024 MAX_COMPLETION_LENGTH=512 MAX_MODEL_LEN=4096 \
GRADIENT_CHECKPOINTING=1 FSDP='full_shard auto_wrap' \
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP=Qwen3DecoderLayer \
REWARD_SERVER_MODE=paired scripts/run_misalignment_ablation.sh
```

Result: vLLM generation and paired reward warmup worked, but GRPO failed inside
PyTorch FSDP during the model forward with
`AssertionError: Non-root FSDP instance's _is_root should not have been set yet`.
DeepSpeed is not installed in this container. The long run therefore uses the
stable DDP path with gradient checkpointing enabled and records FSDP as an
incompatible path for this TRL/vLLM setup.

Long-shape smoke:

```bash
RUN_ID=smoke-longseq-ddp-paired-v1-seed20260428 \
ALGOS=grpo GRPO_COEFS=0 MAX_STEPS=1 \
MAX_PROMPT_LENGTH=2048 MAX_COMPLETION_LENGTH=1024 MAX_MODEL_LEN=4096 \
GRADIENT_ACCUMULATION_STEPS=1 GRADIENT_CHECKPOINTING=1 \
REWARD_SERVER_MODE=paired REWARD_WARMUP_TOKENS=1024 \
scripts/run_misalignment_ablation.sh
```

Result: success. The paired reward warmup took `120.314s` for one
`[1024, 151936]` reward tensor. The one-step GRPO run completed with mean
completion length `791.75`, max length `1024`, `J/token=1.042e-07`, and
`gamma_bracketed_rate=1.0`. Tokenization audit kept prompt-side EOS tokens in
the prompt and found no trailing post-EOS tokens in the completion path.

The 200-step long run uses this stable configuration with GRPO and onlineDPO
baselines plus the smallest previously interesting nonzero coefficient:

| algorithm | J coefficient |
| --- | --- |
| GRPO | `0`, `0.003` |
| onlineDPO | `0`, `0.003` |

Long run status before manual stop:

Command:

```bash
RUN_ID=longseq-200step-paired-v1-seed20260428 \
WANDB_PROJECT=trl-misalignment-longrun-20260428 \
SEED=20260428 \
ALGOS='grpo online_dpo' GRPO_COEFS='0 0.003' ONLINE_DPO_COEFS='0 0.003' \
MAX_STEPS=200 MAX_TRAIN_SAMPLES=4096 \
MAX_PROMPT_LENGTH=2048 MAX_COMPLETION_LENGTH=1024 MAX_MODEL_LEN=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 NUM_GENERATIONS=4 \
LEARNING_RATE=1e-6 LR_SCHEDULER_TYPE=constant_with_warmup WARMUP_RATIO=0.1 \
MAX_GRAD_NORM=10.0 LOGGING_STEPS=1 SAVE_STEPS=100 \
TRAIN_GPUS=0,1,2,3 TRAIN_NUM_PROCESSES=4 VLLM_GPUS=4,5 VLLM_TP=2 \
REWARD_WINNER_GPU=6 REWARD_LOSER_GPU=7 REWARD_SERVER_MODE=paired \
REWARD_WARMUP_TOKENS=1024 GRADIENT_CHECKPOINTING=1 \
scripts/run_misalignment_ablation.sh
```

Artifacts:

- Persistent metadata: `/yj_data/trl_misalignment/longseq-200step-paired-v1-seed20260428`
- Local checkpoints: `/root/trl-misalignment/runs/misalignment_ablation/longseq-200step-paired-v1-seed20260428`
- wandb project: `trl-misalignment-longrun-20260428`

Completed result:

| run | steps | runtime | steps/sec | reward | final J/token | mean J/token | reward scoring/step | step time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO `0` | `200` | `2538.75s` | `0.079` | `12.431` | `8.332e-04` | `8.329e-04` | `3.536s` | `12.454s` |

Interrupted trace:

| run | steps logged | reward | last J/token | mean J/token | last J aux loss | reward scoring/step | step time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO `0.003` | `35` | `-109.229` | `7.780e-04` | `5.893e-04` | `2.390e-03` | `3.566s` | `13.548s` |

The experiment was manually stopped before onlineDPO started. The stopped
process tree was terminated and all GPUs were confirmed free. The partial
`0.003` trace is useful as a smoke/stability check only; it should not be used
as a final ablation comparison against the 200-step baseline.

Operational notes:

- Paired reward warmup for one `[1024, 151936]` reward tensor took `118.703s`.
- The 200-step baseline logged every step, with `gamma_bracketed_rate=1.0`.
- `tokenization/trailing_tokens_after_last_completion_eos` stayed `0.0` in all
  logged rows inspected for both completed and interrupted runs.
- The coefficient run logged `misalignment/J_aux_loss = 0.003 * J`, confirming
  the backpropagated term was active.

### 2026-04-28 Sequence-Level Reward Ambiguity Analysis

Purpose: test whether tokenwise misalignment metrics become unstable or
semantically ambiguous when the winner and loser vocab reward models produce
similar distributions. The concern is that
`softmax(log pi_w - log pi_l)` can be high-entropy and nearly uniform, while J
and `grad J` remain nonzero because the policy moved relative to the reference.

Implementation:

- Added `scripts/analyze_sequence_misalignment.py`, a multi-prompt diagnostic
  that samples prompts, generates completions with the policy checkpoint,
  computes policy/reference full-vocab logits, computes vocab reward vectors,
  and writes tokenwise CSVs plus human-readable plots.
- Added `tests/test_sequence_misalignment_diagnostics.py` for the exact
  constant-reward limit: if reward is constant and policy differs from
  reference, J equals `KL(policy||reference)` and `grad J` is nonzero; if policy
  and reference also match, both vanish.
- Added `matplotlib` for persistent plot generation in the analysis scripts.

Command:

```bash
TRL_EXPERIMENTAL_SILENCE=1 PYTHONUNBUFFERED=1 \
python scripts/analyze_sequence_misalignment.py \
  --policy-model /root/trl-misalignment/runs/misalignment_ablation/longseq-200step-paired-v1-seed20260428/grpo_jcoef_0_seed20260428/checkpoint-200 \
  --reference-model yunjae-won/ubq30i_qwen4b_sft_both \
  --winner-model yunjae-won/ubq30i_qwen4b_sft_yw \
  --loser-model yunjae-won/ubq30i_qwen4b_sft_yl \
  --dataset-name trl-lib/ultrafeedback-prompt \
  --dataset-split train \
  --output-dir /yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428 \
  --seed 20260428 \
  --num-prompts 24 \
  --max-prompt-length 2048 \
  --max-new-tokens 64 \
  --max-analysis-tokens 64 \
  --policy-device cuda:0 \
  --reference-device cuda:1 \
  --reward-winner-device cuda:2 \
  --reward-loser-device cuda:3 \
  --no-reward-compile
```

Artifacts:

- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/report.md`
- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/token_metrics.csv`
- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/sequence_summaries.csv`
- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/plots/aggregate_scatter.png`
- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/plots/entropy_binned_trends.png`
- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/plots/sequence_*.png`

Aggregate result over 24 prompts and 1,536 analyzed tokens:

| metric | mean | median | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| J | `1.121e-03` | `1.999e-04` | `2.540e-03` | `1.363e-01` |
| `grad_J_l2` | `8.688e-03` | `2.455e-03` | `2.457e-02` | `2.077e-01` |
| `KL(policy||ref)` | `1.723e-03` | `3.673e-04` | `3.986e-03` | `3.022e-01` |
| `std(R)` | `5.253e-01` | `4.517e-01` | `7.555e-01` | `4.372e+00` |
| `H(softmax R)/logV` | `9.850e-01` | `9.916e-01` | `9.958e-01` | `9.986e-01` |
| `KL(softmax R||uniform)` | `1.787e-01` | `1.002e-01` | `3.036e-01` | `5.607e+00` |
| `|gamma|*std(R)` | `3.472e-02` | `1.865e-02` | `8.489e-02` | `4.619e-01` |

Interpretation:

- The computation is numerically stable in the low-signal limit, but the metric
  changes meaning. When `R = log pi_w - log pi_l` is constant, the exponential
  tilt cannot use the reward direction, so J collapses to
  `KL(policy||reference)`.
- In the GRPO step-200 scan, strict high-entropy/near-uniform reward rows were
  rare by the combined threshold `H/logV >= 0.995` and
  `KL(softmax R||uniform) <= 0.02` (`1/1536` rows). Low-scale rows were common:
  `982/1536` rows had `std(R) <= 0.5`.
- The highest-J tokens were not primarily the near-uniform reward cases. They
  had large policy/reference KL and more non-uniform reward vectors. The top
  token was EOS with `J=0.136`, `grad_J_l2=0.194`,
  `KL(policy||ref)=0.302`, `std(R)=1.282`, and
  `KL(softmax R||uniform)=0.801`.
- `J` tracked `KL(policy||ref)` much more strongly than reward-vector shape in
  this checkpoint. Reward shape modulated the severity, while policy/reference
  drift was the dominant factor.
- Normalized reward entropy alone is a weak reliability test for a large
  vocabulary. Use it together with `KL(softmax R||uniform)`, `std(R)`, and
  `|gamma|*std(R)`.

Practical consequence:

- Paper plots should stratify J and `grad J` by reward-vector reliability.
- Low-std or near-uniform reward rows should be reported as ambiguous residual
  policy/reference drift, not as clean reward-exploitation evidence.
- For auxiliary-loss ablations, add a reliability-weighted variant such as
  `J * clip(KL(softmax R||uniform) / tau, 0, 1)` or a low-std mask, while
  retaining the original unweighted J as the primary diagnostic.

## Artifacts

Each run group writes:

- `run_plan.txt`: fixed hyperparameters and storage layout
- `manifest.jsonl`: one row per experiment
- `logs/*.log`: reward server, vLLM, and training logs
- `tokenization_debug/*.jsonl`: prompt and first-rollout tokenization samples
- `summary_metrics.csv`: final metric row per experiment
- `analysis.csv`: baseline deltas and selected metrics
- `analysis.md`: human-readable comparison table

wandb contains the time-series curves. The CSV/Markdown files are the compact
paper-analysis entrypoints.
