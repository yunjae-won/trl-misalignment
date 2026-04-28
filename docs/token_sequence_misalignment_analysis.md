# Token Sequence Misalignment Analysis

This note documents the diagnostic added in
`scripts/analyze_token_sequence_misalignment.py`. It is meant for the failure
mode where the two vocab-level reward models are nearly tied:

```text
R_t(v) = log pi_w(v | prefix_t) - log pi_l(v | prefix_t)
```

If `pi_w` and `pi_l` are similar, `R_t` can have low magnitude and its
direction can be dominated by estimation noise. The misalignment metric should
therefore be interpreted together with reward-signal diagnostics.

## What The Tool Computes

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
