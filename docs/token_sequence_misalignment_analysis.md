# Token Sequence Misalignment Analysis

This note documents the sequence diagnostics added in
`scripts/analyze_token_sequence_misalignment.py` and
`scripts/analyze_sequence_misalignment.py`. They are meant for the failure mode
where the two vocab-level reward models are nearly tied:

```text
R_t(v) = log pi_w(v | prefix_t) - log pi_l(v | prefix_t)
```

If `pi_w` and `pi_l` are similar, `R_t` can have low magnitude and its
direction can be dominated by estimation noise. The misalignment metric should
therefore be interpreted together with reward-signal diagnostics.

## What The Single-Sequence Tool Computes

For one prompt/completion sequence, the script writes:

- `per_token_metrics.csv`: one row per completion token.
- `summary.json`: aggregate weak-vs-strong reward signal statistics.
- `token_misalignment_dashboard.svg`: dependency-free trend dashboard.
- `report.md`: prompt, completion, summary, and highest-J positions.
- `completion_ids.json`: exact analyzed sequence for reuse across checkpoints.

Each token row includes:

- policy/reference metrics: `J`, `reverse_kl_divergence`, `entropy_a`,
  `entropy_b`, `gamma_star`, `gamma_bracketed`
- reward-vector strength: `reward_vocab_std`, `reward_vocab_range`,
  `reward_vocab_abs_max`
- winner/loser similarity: `winner_loser_js`, `winner_loser_tv`,
  `winner_loser_top1_agreement`
- sampled-token diagnostics: `selected_reward`,
  `selected_reward_percentile`, selected policy/reference probabilities
- a perturbation probe: `J_noise_mean`, `J_noise_std`, `J_noise_cv` after
  adding small absolute Gaussian noise to the centered reward vector

The `weak_reward_signal` column marks the bottom quantile of
`reward_vocab_std` within the analyzed sequence. It is a local reliability
flag, not a universal threshold.

## What The Aggregate Tool Computes

`scripts/analyze_sequence_misalignment.py` samples many prompts, generates
policy completions, and writes tokenwise diagnostics for each generated
completion. It uses the same chat-template tokenization path as training and
normalizes only the completion IDs before reward scoring. This preserves
prompt-side EOS tokens while removing post-EOS completion tokens when they are
present.

The aggregate output directory contains:

- `token_metrics.csv`: one row per analyzed completion token.
- `sequence_summaries.csv`: one row per sampled prompt/completion sequence.
- `config.json`: exact command-line configuration.
- `report.md`: aggregate tables, highest-J rows, and interpretation guide.
- `plots/aggregate_scatter.png`: reward entropy, reward non-uniformity,
  reward scale, J, grad J, and policy/reference KL in one dashboard.
- `plots/entropy_binned_trends.png`: token rows sorted by reward entropy and
  binned into trend lines.
- `plots/sequence_*.png`: per-token traces with token labels on the x-axis.

Additional columns beyond the single-sequence tool include:

- `grad_J_l2`, `grad_J_linf`: autograd norm of J with respect to policy logits.
- `reward_softmax_entropy_norm`: `H(softmax(R)) / log(V)`.
- `reward_softmax_kl_uniform`: `KL(softmax(R) || uniform)`, the preferred
  companion to entropy because normalized entropy is often close to one for a
  large vocabulary.
- `reward_softmax_effective_vocab_frac`: `exp(H(softmax(R))) / V`.
- `gamma_abs_times_reward_std`: effective tilt scale, useful when gamma is
  large but the reward vector is small.
- `selected_token_reward_percentile`: percentile rank of the actually sampled
  token under the full reward vector.
- `reward_logratio_corr_uniform` and `reward_logratio_corr_reference`:
  correlations between reward direction and policy/reference log-ratio under
  uniform and reference weighting.

## Theory Note

The tokenwise `J` currently used here is:

1. Center the reward vector `R_t`.
2. Find the exponential tilt of the reference distribution,
   `q_gamma(v) ∝ pi_ref(v) exp(gamma R_centered(v))`, whose expected reward
   matches the policy expected reward.
3. Report `KL(pi_policy || q_gamma)`.

Consequences:

- If `R_t` is exactly constant, there is no reward direction. The solver sets
  `gamma=0`, `q_gamma=pi_ref`, and `J = KL(pi_policy || pi_ref)`.
- If `R_t` is tiny but nonconstant, `J` is mostly scale-invariant: multiplying
  `R_t` by a positive scalar changes `gamma` inversely but leaves
  `gamma * R_t` nearly unchanged.
- Therefore a small `log pi_w - log pi_l` magnitude does not necessarily make
  `J` small. It makes the semantic interpretation of the reward direction less
  reliable.
- In weak-reward regions, high `J` often means: “the policy moved relative to
  the reference in a way not explained by the weak/noisy reward direction,”
  not necessarily “the reward model found a strong misalignment signal.”

The exact constant-reward limit is now covered by
`tests/test_sequence_misalignment_diagnostics.py`. If the reward vector is
constant and the policy differs from the reference, `J` equals
`KL(pi_policy || pi_ref)` and `grad_J` is nonzero. If the policy and reference
are also identical, both `J` and `grad_J` vanish. This means the computation is
stable, but not automatically semantically reliable in low-signal reward
regions.

For paper-facing analyses, report `J` together with `reward_vocab_std` or
`winner_loser_js`, and consider filtering or stratifying low-signal tokens.

## Example Runs

Both runs used dataset index `5000`, outside the first `4096` prompts used by
the long-run training job, and analyzed the same 64 generated completion tokens.

```bash
python scripts/analyze_token_sequence_misalignment.py \
  --policy-model /root/trl-misalignment/runs/misalignment_ablation/longseq-200step-paired-v1-seed20260428/grpo_jcoef_0_seed20260428/checkpoint-200 \
  --reference-model yunjae-won/ubq30i_qwen4b_sft_both \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --sample-index 5000 \
  --max-new-tokens 96 \
  --max-positions 64 \
  --noise-std 0.001 \
  --noise-samples 6 \
  --run-name longseq-grpo0-ckpt200-sample5000-64tok
```

```bash
python scripts/analyze_token_sequence_misalignment.py \
  --policy-model /root/trl-misalignment/runs/misalignment_ablation/longseq-200step-paired-v1-seed20260428/grpo_jcoef_0_seed20260428/checkpoint-100 \
  --reference-model yunjae-won/ubq30i_qwen4b_sft_both \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
  --sample-index 5000 \
  --completion-ids-json /yj_data/trl_misalignment/token_sequence_analysis/longseq-grpo0-ckpt200-sample5000-64tok/completion_ids.json \
  --max-positions 64 \
  --noise-std 0.001 \
  --noise-samples 6 \
  --run-name longseq-grpo0-ckpt100-sample5000-64tok
```

Artifacts:

- `/yj_data/trl_misalignment/token_sequence_analysis/longseq-grpo0-ckpt200-sample5000-64tok`
- `/yj_data/trl_misalignment/token_sequence_analysis/longseq-grpo0-ckpt100-sample5000-64tok`

Summary:

| policy checkpoint | mean J | weak-token mean J | strong-token mean J | mean KL(policy||ref) | mean reward std | mean winner/loser JS | mean noise CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO step 100 | `1.132e-03` | `9.403e-04` | `1.196e-03` | `1.621e-03` | `0.538` | `2.455e-02` | `2.949e-02` |
| GRPO step 200 | `1.066e-03` | `9.733e-04` | `1.097e-03` | `1.605e-03` | `0.538` | `2.455e-02` | `4.407e-02` |

Interpretation:

- The analyzed sequence does not show a global instability: `gamma_bracketed`
  was `1.0` for both checkpoints, and the absolute-noise probe had low mean CV.
- Weak reward tokens had lower average `J` than strong reward tokens, but this
  is not guaranteed token-by-token.
- The highest-`J` token at step 200 was position `5`, token `" technology"`:
  `J=7.45e-03`, `reward_vocab_std=0.382`, and `winner_loser_js=3.95e-03`.
  This is exactly the cautionary case: winner and loser reward distributions
  are close, yet policy/reference divergence remains high, so the measured
  misalignment is mostly a residual divergence relative to a weak reward
  direction.
- A near-deterministic low-signal token, position `44` (`"ificial"`), had
  `reward_vocab_std=0.251`, `winner_loser_js=5.85e-08`, and essentially zero
  `J`. Its `J_noise_cv` is large because the denominator is near zero; the
  absolute instability is negligible.

Practical recommendation: use `J` as the main objective metric only after
stratifying or weighting by reward signal strength. Low `reward_vocab_std` or
low `winner_loser_js` tokens should be reported separately because they measure
policy/reference residual movement under an unreliable reward direction.

## Aggregate Scan: GRPO Step 200

The first multi-sequence scan used the completed long-run GRPO baseline
checkpoint and 24 train-set prompts, with seed `20260428`.

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
- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/plots/aggregate_scatter.png`
- `/yj_data/trl_misalignment/sequence-diagnostic-grpo200-train24-seed20260428/plots/entropy_binned_trends.png`

Summary over 1,536 analyzed tokens:

| metric | mean | median | p90 | max |
| --- | ---: | ---: | ---: | ---: |
| `J` | `1.121e-03` | `1.999e-04` | `2.540e-03` | `1.363e-01` |
| `grad_J_l2` | `8.688e-03` | `2.455e-03` | `2.457e-02` | `2.077e-01` |
| `KL(policy||ref)` | `1.723e-03` | `3.673e-04` | `3.986e-03` | `3.022e-01` |
| `std(R)` | `5.253e-01` | `4.517e-01` | `7.555e-01` | `4.372e+00` |
| `H(softmax R)/logV` | `9.850e-01` | `9.916e-01` | `9.958e-01` | `9.986e-01` |
| `KL(softmax R||uniform)` | `1.787e-01` | `1.002e-01` | `3.036e-01` | `5.607e+00` |
| `|gamma|*std(R)` | `3.472e-02` | `1.865e-02` | `8.489e-02` | `4.619e-01` |

Pattern:

- Strictly near-uniform reward rows were rare: `1/1536` rows met
  `H/logV >= 0.995` and `KL(softmax R||uniform) <= 0.02`.
- Low-reward-scale rows were common: `982/1536` rows had `std(R) <= 0.5`.
- High-entropy and low-std rows had lower average J and grad J than the
  non-uniform reward tail, but they still produced nonzero values when the
  policy had drifted from the reference.
- The largest-J rows were dominated by large `KL(policy||ref)` and more
  non-uniform reward shapes. The top row was an EOS token with
  `J=0.136`, `grad_J_l2=0.194`, `KL(policy||ref)=0.302`, `std(R)=1.282`,
  and `KL(softmax R||uniform)=0.801`.
- Across all rows, `J` correlated far more with `KL(policy||ref)` than with
  reward-shape statistics. This supports reading high J as residual
  policy/reference movement first, and reward exploitation only after checking
  the reward vector has a meaningful shape.

Paper-facing recommendation:

- Report `J` and `grad_J` stratified by `std(R)` and
  `KL(softmax R||uniform)`.
- Treat rows with high entropy, low KL-to-uniform, or low reward std as
  ambiguous. They are useful evidence about optimizer pressure, but not clean
  evidence that the reward model exposed a meaningful token-level preference.
- Consider an ambiguity-weighted metric such as
  `J * clip(KL(softmax R||uniform) / tau, 0, 1)` for auxiliary training or
  robustness checks. The unweighted J should remain logged so the paper can
  distinguish the original metric from the reliability-weighted variant.
