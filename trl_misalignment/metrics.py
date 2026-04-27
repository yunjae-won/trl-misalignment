from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass
class VocabMisalignmentConfig:
    """Runtime options for vocab-level misalignment monitoring.

    `backprop_j=False` is the monitoring setting. In that mode all inputs are
    detached before metric computation. If `backprop_j=True`, only the policy
    tensor `A` remains attached; `B`, `R`, and the gamma solve are detached.
    """

    enabled: bool = True
    beta: Union[float, torch.Tensor] = 0.0
    normalize_inputs: bool = True
    compute_dtype: Optional[torch.dtype] = torch.float64
    gamma_iters: int = 80
    bracket_iters: int = 20
    initial_logit_span: float = 64.0
    reward_tol: Optional[float] = None
    ppo_clip_eps: Optional[float] = 0.2
    backprop_j: bool = False
    prompt_reduce: str = "sum"
    log_prefix: str = "misalignment"
    log_keys: Sequence[str] = field(
        default_factory=lambda: (
            "J",
            "reward_a",
            "reward_b",
            "reward_improvement",
            "reverse_kl_divergence",
            "forward_kl_divergence",
            "rlhf_score",
            "entropy_a",
            "entropy_b",
            "symmetric_kl",
            "js_divergence",
            "tv_distance",
            "gamma_star",
            "gamma_reward_residual",
        )
    )


def batched_vocablevel_misalignment(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    beta: Union[float, torch.Tensor],
    *,
    normalize_inputs: bool = True,
    compute_dtype: Optional[torch.dtype] = torch.float64,
    gamma_iters: int = 80,
    bracket_iters: int = 20,
    initial_logit_span: float = 64.0,
    reward_tol: Optional[float] = None,
    ppo_clip_eps: Optional[float] = 0.2,
) -> Dict[str, torch.Tensor]:
    """Compute per-row misalignment metrics for batched categorical distributions."""

    if A.shape != B.shape or A.shape != R.shape:
        raise ValueError(
            "A, B, and R must have the same shape; got "
            f"{A.shape}, {B.shape}, {R.shape}."
        )
    if A.ndim != 2:
        raise ValueError(f"Expected shape [batch, vocab], got {tuple(A.shape)}.")
    if not torch.is_floating_point(A) or not torch.is_floating_point(B) or not torch.is_floating_point(R):
        raise TypeError("A, B, and R must be floating-point tensors.")
    if A.device != B.device or A.device != R.device:
        raise ValueError("A, B, and R must be on the same device.")

    device = A.device

    if compute_dtype is None:
        work_dtype = torch.promote_types(torch.promote_types(A.dtype, B.dtype), R.dtype)
        if work_dtype in (torch.float16, torch.bfloat16):
            work_dtype = torch.float32
    else:
        work_dtype = compute_dtype

    A_ = A.to(work_dtype)
    B_ = B.to(work_dtype)
    R_ = R.to(work_dtype)

    if normalize_inputs:
        log_a = F.log_softmax(A_, dim=-1)
        log_b = F.log_softmax(B_, dim=-1)
    else:
        log_a = A_
        log_b = B_

    p_a = F.softmax(A_, dim=-1)
    p_b = F.softmax(B_, dim=-1)

    def kl_logps(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
        p = logp.exp()
        diff = logp - logq
        diff = torch.where(p > 0, diff, 0)
        return (p * diff).sum(dim=-1)

    def entropy_from_logp(logp: torch.Tensor) -> torch.Tensor:
        p = logp.exp()
        safe_logp = torch.where(p > 0, logp, 0)
        return -(p * safe_logp).sum(dim=-1)

    reward_a = (p_a * R_).sum(dim=-1)
    reward_b = (p_b * R_).sum(dim=-1)
    reward_improvement = reward_a - reward_b

    reverse_kl = kl_logps(log_a, log_b)
    forward_kl = kl_logps(log_b, log_a)

    beta_t = torch.as_tensor(beta, dtype=work_dtype, device=device)
    rlhf_score = reward_a - beta_t * reverse_kl

    entropy_a = entropy_from_logp(log_a)
    entropy_b = entropy_from_logp(log_b)

    cross_entropy_a_b = entropy_a + reverse_kl
    cross_entropy_b_a = entropy_b + forward_kl

    log_2 = torch.log(torch.tensor(2.0, dtype=work_dtype, device=device))
    log_m = torch.logaddexp(log_a, log_b) - log_2
    js_divergence = 0.5 * kl_logps(log_a, log_m) + 0.5 * kl_logps(log_b, log_m)

    symmetric_kl = reverse_kl + forward_kl
    tv_distance = 0.5 * (p_a - p_b).abs().sum(dim=-1)

    reward_var_a = (p_a * (R_ - reward_a[:, None]).square()).sum(dim=-1)
    reward_var_b = (p_b * (R_ - reward_b[:, None]).square()).sum(dim=-1)

    log_ratio = log_a - log_b
    log_ratio_mean_under_a = (p_a * torch.where(p_a > 0, log_ratio, 0)).sum(dim=-1)
    log_ratio_mean_under_b = (p_b * torch.where(p_b > 0, log_ratio, 0)).sum(dim=-1)

    out: Dict[str, torch.Tensor] = {
        "reward_a": reward_a,
        "reward_b": reward_b,
        "reward_improvement": reward_improvement,
        "reverse_kl_divergence": reverse_kl,
        "forward_kl_divergence": forward_kl,
        "rlhf_score": rlhf_score,
        "entropy_a": entropy_a,
        "entropy_b": entropy_b,
        "cross_entropy_a_b": cross_entropy_a_b,
        "cross_entropy_b_a": cross_entropy_b_a,
        "symmetric_kl": symmetric_kl,
        "js_divergence": js_divergence,
        "tv_distance": tv_distance,
        "reward_var_a": reward_var_a,
        "reward_var_b": reward_var_b,
        "log_ratio_mean_under_a": log_ratio_mean_under_a,
        "log_ratio_mean_under_b": log_ratio_mean_under_b,
    }

    if ppo_clip_eps is not None:
        if not (0.0 < float(ppo_clip_eps) < 1.0):
            raise ValueError("ppo_clip_eps must be in (0, 1), or None.")

        hi = torch.log1p(torch.tensor(float(ppo_clip_eps), dtype=work_dtype, device=device))
        lo = torch.log1p(torch.tensor(-float(ppo_clip_eps), dtype=work_dtype, device=device))

        clipped = (log_ratio > hi) | (log_ratio < lo)
        clipped_f = clipped.to(work_dtype)

        out["ppo_clip_fraction_under_b"] = (p_b * clipped_f).sum(dim=-1)
        out["ppo_clip_fraction_under_a"] = (p_a * clipped_f).sum(dim=-1)
        out["ppo_exact_old_new_kl"] = forward_kl
        out["log_ratio_var_under_b"] = (
            p_b * (log_ratio - log_ratio_mean_under_b[:, None]).square()
        ).sum(dim=-1)

    r_center = R_ - R_.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def tilted_reward_centered(
        gamma: torch.Tensor,
        log_b_ng: torch.Tensor,
        r_c_ng: torch.Tensor,
    ) -> torch.Tensor:
        logits = log_b_ng + gamma[:, None] * r_c_ng
        q = F.softmax(logits, dim=-1)
        return (q * r_c_ng).sum(dim=-1)

    @torch.no_grad()
    def solve_gamma() -> Tuple[torch.Tensor, torch.Tensor]:
        log_b_ng = log_b.detach()
        log_a_ng = log_a.detach()
        r_c_ng = r_center.detach()

        target_ng = (log_a_ng.exp() * r_c_ng).sum(dim=-1)

        r_min = r_c_ng.min(dim=-1).values
        r_max = r_c_ng.max(dim=-1).values
        r_range = r_max - r_min

        if reward_tol is None:
            tol = 1e-12 if work_dtype == torch.float64 else 1e-6
        else:
            tol = float(reward_tol)

        active = r_range > tol
        init = torch.as_tensor(initial_logit_span, dtype=work_dtype, device=device) / r_range.clamp_min(tol)

        lower = -init
        upper = init

        for _ in range(bracket_iters):
            f_low = tilted_reward_centered(lower, log_b_ng, r_c_ng) - target_ng
            f_high = tilted_reward_centered(upper, log_b_ng, r_c_ng) - target_ng

            need_lower = active & (f_low > 0)
            need_upper = active & (f_high < 0)

            if not bool((need_lower | need_upper).any()):
                break

            lower = torch.where(need_lower, lower * 2.0, lower)
            upper = torch.where(need_upper, upper * 2.0, upper)

        f_low = tilted_reward_centered(lower, log_b_ng, r_c_ng) - target_ng
        f_high = tilted_reward_centered(upper, log_b_ng, r_c_ng) - target_ng

        bracketed = (~active) | ((f_low <= 0) & (f_high >= 0))
        solve_mask = active & bracketed

        lo = lower.clone()
        hi = upper.clone()

        for _ in range(gamma_iters):
            mid = 0.5 * (lo + hi)
            f_mid = tilted_reward_centered(mid, log_b_ng, r_c_ng) - target_ng

            go_right = solve_mask & (f_mid < 0)
            go_left = solve_mask & ~go_right

            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_left, mid, hi)

        gamma = 0.5 * (lo + hi)
        gamma = torch.where(active, gamma, 0)
        return gamma.detach(), bracketed.detach()

    gamma_star, gamma_bracketed = solve_gamma()
    gamma_star = gamma_star.to(work_dtype).detach()

    tilted_logits = log_b + gamma_star[:, None] * r_center
    tilted_log_normalizer = torch.logsumexp(tilted_logits, dim=-1)
    log_q_gamma = tilted_logits - tilted_log_normalizer[:, None]

    J = kl_logps(log_a, log_q_gamma)

    q_gamma = log_q_gamma.exp()
    tilted_reward = (q_gamma * R_).sum(dim=-1)
    gamma_reward_residual = tilted_reward - reward_a

    out.update(
        {
            "gamma_star": gamma_star,
            "J": J,
            "tilted_reward": tilted_reward,
            "tilted_log_normalizer": tilted_log_normalizer,
            "gamma_reward_residual": gamma_reward_residual,
            "gamma_bracketed": gamma_bracketed,
        }
    )

    return out


def _flatten_completion_rows(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    completion_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if A.shape != B.shape or A.shape != R.shape:
        raise ValueError(f"A, B, and R must share shape [batch, time, vocab], got {A.shape}, {B.shape}, {R.shape}.")
    if A.ndim != 3:
        raise ValueError(f"Expected A/B/R to have shape [batch, time, vocab], got {tuple(A.shape)}.")
    if completion_mask.shape != A.shape[:2]:
        raise ValueError(
            "completion_mask must have shape [batch, time], got "
            f"{tuple(completion_mask.shape)} for tensor shape {tuple(A.shape)}."
        )
    mask = completion_mask.bool()
    batch_ids = torch.arange(A.shape[0], device=A.device).unsqueeze(1).expand_as(mask)[mask]
    lengths = mask.sum(dim=1)
    return A[mask], B[mask], R[mask], batch_ids, lengths


def completion_vocab_misalignment(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    completion_mask: torch.Tensor,
    config: VocabMisalignmentConfig,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute token-row and prompt-aggregated vocab misalignment.

    Args:
        A, B, R: tensors shaped `[batch, completion_time, vocab]`.
        completion_mask: `1` for valid completion tokens, `0` for padding or
            externally-generated/tool tokens.
        config: misalignment options.

    Returns:
        `{"token": ..., "prompt": ..., "lengths": ...}`. Prompt metrics are
        summed per prompt by default, matching `torch.segment_reduce(..., sum)`
        on flattened valid completion rows.
    """

    if not config.enabled:
        return {"token": {}, "prompt": {}, "lengths": completion_mask.sum(dim=1)}

    flat_a, flat_b, flat_r, batch_ids, lengths = _flatten_completion_rows(A, B, R, completion_mask)
    if flat_a.numel() == 0:
        return {"token": {}, "prompt": {}, "lengths": lengths}

    if config.backprop_j:
        metric_inputs = (flat_a, flat_b.detach(), flat_r.detach())
        token_metrics = batched_vocablevel_misalignment(
            *metric_inputs,
            beta=config.beta,
            normalize_inputs=config.normalize_inputs,
            compute_dtype=config.compute_dtype,
            gamma_iters=config.gamma_iters,
            bracket_iters=config.bracket_iters,
            initial_logit_span=config.initial_logit_span,
            reward_tol=config.reward_tol,
            ppo_clip_eps=config.ppo_clip_eps,
        )
    else:
        with torch.no_grad():
            token_metrics = batched_vocablevel_misalignment(
                flat_a.detach(),
                flat_b.detach(),
                flat_r.detach(),
                beta=config.beta,
                normalize_inputs=config.normalize_inputs,
                compute_dtype=config.compute_dtype,
                gamma_iters=config.gamma_iters,
                bracket_iters=config.bracket_iters,
                initial_logit_span=config.initial_logit_span,
                reward_tol=config.reward_tol,
                ppo_clip_eps=config.ppo_clip_eps,
            )

    prompt_metrics: Dict[str, torch.Tensor] = {}
    for key, values in token_metrics.items():
        if values.dtype == torch.bool:
            values_f = values.float()
        else:
            values_f = values
        if values_f.ndim != 1:
            continue
        reduced = values_f.new_zeros((A.shape[0],), dtype=values_f.dtype)
        reduced.index_add_(0, batch_ids, values_f)
        if config.prompt_reduce == "mean":
            reduced = reduced / lengths.to(reduced.dtype).clamp_min(1)
        elif config.prompt_reduce != "sum":
            raise ValueError(f"Unsupported prompt_reduce={config.prompt_reduce!r}; use 'sum' or 'mean'.")
        prompt_metrics[key] = reduced

    return {"token": token_metrics, "prompt": prompt_metrics, "lengths": lengths}


def detached_scalar_logs(
    prompt_metrics: Mapping[str, torch.Tensor],
    *,
    prefix: str,
    keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Convert prompt-level tensors into mean scalar tensors for trainer logs."""

    logs: Dict[str, torch.Tensor] = {}
    for key in keys:
        value = prompt_metrics.get(key)
        if value is None or value.numel() == 0:
            continue
        if value.dtype == torch.bool:
            value = value.float()
        logs[f"{prefix}/{key}"] = value.detach().float().mean()
    return logs
