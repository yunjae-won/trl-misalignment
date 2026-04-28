from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_misalignment.compat import apply_runtime_compatibility_patches
from trl_misalignment.metrics import batched_vocablevel_misalignment
from trl_misalignment.rewards import TokenVocabRewardProvider


apply_runtime_compatibility_patches()


DEFAULT_POLICY = "yunjae-won/ubq30i_qwen4b_sft_both"
DEFAULT_WINNER = "yunjae-won/ubq30i_qwen4b_sft_yw"
DEFAULT_LOSER = "yunjae-won/ubq30i_qwen4b_sft_yl"
DEFAULT_DATASET = "trl-lib/ultrafeedback-prompt"


@dataclass
class SequenceResult:
    sequence_id: int
    prompt_index: int
    prompt_text: str
    completion_text: str
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate token-level diagnostics for vocab-reward misalignment. "
            "The tool samples prompts, generates completions with a policy, "
            "computes policy/reference/reward full-vocab rows, and writes "
            "CSV, plots, and a Markdown report."
        )
    )
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--reference-model", default=DEFAULT_POLICY)
    parser.add_argument("--winner-model", default=DEFAULT_WINNER)
    parser.add_argument("--loser-model", default=DEFAULT_LOSER)
    parser.add_argument("--tokenizer-fallback", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-analysis-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:1")
    parser.add_argument("--reward-winner-device", default="cuda:2")
    parser.add_argument("--reward-loser-device", default="cuda:3")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--reward-logprob-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--no-reward-compile", action="store_true")
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--compute-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--plot-top-sequences", type=int, default=6)
    parser.add_argument("--high-entropy-threshold", type=float, default=0.995)
    parser.add_argument("--low-reward-kl-threshold", type=float, default=0.02)
    parser.add_argument("--low-reward-std-threshold", type=float, default=0.5)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def metric_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def load_tokenizer(model: str, fallback: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", truncation_side="left")
    except Exception as exc:
        print(f"[tokenizer] failed to load {model!r}: {exc}. Falling back to {fallback!r}.", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(fallback, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model: str, *, dtype: torch.dtype, device: str):
    loaded = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, low_cpu_mem_usage=True)
    loaded.to(device)
    loaded.eval()
    for param in loaded.parameters():
        param.requires_grad_(False)
    return loaded


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return prompt
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def prompt_to_text(prompt: list[dict[str, str]]) -> str:
    return "\n".join(f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in prompt)


def prompt_ids_from_messages(tokenizer, prompt: list[dict[str, str]], max_prompt_length: int) -> list[int]:
    ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    if len(ids) > max_prompt_length:
        ids = ids[-max_prompt_length:]
    return [int(x) for x in ids]


@torch.inference_mode()
def generate_completion(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[int]:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})
    output_ids = model.generate(**kwargs)[0].detach().cpu().tolist()
    return [int(x) for x in output_ids[len(prompt_ids) :]]


@torch.inference_mode()
def completion_logits(
    model,
    *,
    prompt_ids: list[int],
    completion_ids: list[int],
    device: str,
) -> torch.Tensor:
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    logits = logits[:, :-1, :]
    logits = logits[:, -len(completion_ids) :, :].squeeze(0)
    return logits.float()


def weighted_corr(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    x_mean = (weights * x).sum(dim=-1, keepdim=True)
    y_mean = (weights * y).sum(dim=-1, keepdim=True)
    x_c = x - x_mean
    y_c = y - y_mean
    cov = (weights * x_c * y_c).sum(dim=-1)
    x_var = (weights * x_c.square()).sum(dim=-1)
    y_var = (weights * y_c.square()).sum(dim=-1)
    denom = (x_var * y_var).sqrt().clamp_min(torch.finfo(weights.dtype).eps)
    return cov / denom


def token_diagnostics(
    *,
    tokenizer,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    rewards: torch.Tensor,
    completion_ids: list[int],
    beta: float,
    compute_dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not (policy_logits.shape == reference_logits.shape == rewards.shape):
        raise ValueError(
            "policy_logits, reference_logits, and rewards must share shape; got "
            f"{tuple(policy_logits.shape)}, {tuple(reference_logits.shape)}, {tuple(rewards.shape)}"
        )

    device = policy_logits.device
    vocab = policy_logits.shape[-1]
    log_vocab = math.log(vocab)

    A = policy_logits.detach().clone().requires_grad_(True)
    B = reference_logits.detach().to(device)
    R = rewards.detach().to(device)
    metrics = batched_vocablevel_misalignment(
        A,
        B,
        R,
        beta=beta,
        compute_dtype=compute_dtype,
        normalize_inputs=True,
    )
    grad = torch.autograd.grad(metrics["J"].sum(), A, retain_graph=False)[0].detach()

    with torch.no_grad():
        log_a = F.log_softmax(policy_logits.detach(), dim=-1)
        log_b = F.log_softmax(reference_logits.detach().to(device), dim=-1)
        p_a = log_a.exp()
        p_b = log_b.exp()
        log_ratio = log_a - log_b

        reward_log_probs = F.log_softmax(R, dim=-1)
        reward_probs = reward_log_probs.exp()
        reward_entropy = -(reward_probs * reward_log_probs).sum(dim=-1)
        reward_entropy_norm = reward_entropy / log_vocab
        reward_kl_uniform = log_vocab - reward_entropy
        reward_kl_uniform_norm = reward_kl_uniform / log_vocab
        reward_effective_vocab_frac = reward_entropy.exp() / vocab
        reward_max_prob = reward_probs.max(dim=-1).values
        reward_top_ids = R.argmax(dim=-1)
        reward_top_prob_policy = p_a.gather(1, reward_top_ids[:, None]).squeeze(1)
        reward_top_prob_reference = p_b.gather(1, reward_top_ids[:, None]).squeeze(1)

        selected_ids = torch.tensor(completion_ids, dtype=torch.long, device=device)
        selected_reward = R.gather(1, selected_ids[:, None]).squeeze(1)
        selected_reward_percentile = (R <= selected_reward[:, None]).float().mean(dim=-1)
        selected_reward_softmax_prob = reward_probs.gather(1, selected_ids[:, None]).squeeze(1)
        selected_policy_prob = p_a.gather(1, selected_ids[:, None]).squeeze(1)
        selected_reference_prob = p_b.gather(1, selected_ids[:, None]).squeeze(1)

        uniform = torch.full_like(p_b, 1.0 / vocab)
        reward_logratio_corr_uniform = weighted_corr(R, log_ratio, uniform)
        reward_logratio_corr_reference = weighted_corr(R, log_ratio, p_b)
        gamma_reward_std = metrics["gamma_star"].abs() * metrics["reward_vocab_std"]
        grad_l2 = grad.square().sum(dim=-1).sqrt()
        grad_linf = grad.abs().max(dim=-1).values

        rows: list[dict[str, Any]] = []
        for t, token_id in enumerate(completion_ids):
            token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            reward_top_text = tokenizer.decode([int(reward_top_ids[t].item())], skip_special_tokens=False)
            rows.append(
                {
                    "step": t,
                    "token_id": int(token_id),
                    "token_text": printable_token(token_text),
                    "reward_top_token_id": int(reward_top_ids[t].item()),
                    "reward_top_token_text": printable_token(reward_top_text),
                    "J": tensor_item(metrics["J"][t]),
                    "grad_J_l2": tensor_item(grad_l2[t]),
                    "grad_J_linf": tensor_item(grad_linf[t]),
                    "gamma_star": tensor_item(metrics["gamma_star"][t]),
                    "gamma_abs_times_reward_std": tensor_item(gamma_reward_std[t]),
                    "reverse_kl": tensor_item(metrics["reverse_kl_divergence"][t]),
                    "forward_kl": tensor_item(metrics["forward_kl_divergence"][t]),
                    "js_divergence": tensor_item(metrics["js_divergence"][t]),
                    "tv_distance": tensor_item(metrics["tv_distance"][t]),
                    "entropy_policy": tensor_item(metrics["entropy_a"][t]),
                    "entropy_reference": tensor_item(metrics["entropy_b"][t]),
                    "reward_a": tensor_item(metrics["reward_a"][t]),
                    "reward_b": tensor_item(metrics["reward_b"][t]),
                    "reward_improvement": tensor_item(metrics["reward_improvement"][t]),
                    "reward_vocab_mean": tensor_item(metrics["reward_vocab_mean"][t]),
                    "reward_vocab_std": tensor_item(metrics["reward_vocab_std"][t]),
                    "reward_vocab_range": tensor_item(metrics["reward_vocab_range"][t]),
                    "reward_vocab_abs_max": tensor_item(metrics["reward_vocab_abs_max"][t]),
                    "reward_softmax_entropy": tensor_item(reward_entropy[t]),
                    "reward_softmax_entropy_norm": tensor_item(reward_entropy_norm[t]),
                    "reward_softmax_kl_uniform": tensor_item(reward_kl_uniform[t]),
                    "reward_softmax_kl_uniform_norm": tensor_item(reward_kl_uniform_norm[t]),
                    "reward_softmax_effective_vocab_frac": tensor_item(reward_effective_vocab_frac[t]),
                    "reward_softmax_max_prob": tensor_item(reward_max_prob[t]),
                    "reward_top_policy_prob": tensor_item(reward_top_prob_policy[t]),
                    "reward_top_reference_prob": tensor_item(reward_top_prob_reference[t]),
                    "selected_token_reward": tensor_item(selected_reward[t]),
                    "selected_token_reward_percentile": tensor_item(selected_reward_percentile[t]),
                    "selected_token_reward_softmax_prob": tensor_item(selected_reward_softmax_prob[t]),
                    "selected_token_policy_prob": tensor_item(selected_policy_prob[t]),
                    "selected_token_reference_prob": tensor_item(selected_reference_prob[t]),
                    "reward_logratio_corr_uniform": tensor_item(reward_logratio_corr_uniform[t]),
                    "reward_logratio_corr_reference": tensor_item(reward_logratio_corr_reference[t]),
                }
            )

    summary = summarize_rows(rows)
    return rows, summary


def tensor_item(x: torch.Tensor) -> float:
    return float(x.detach().float().cpu().item())


def printable_token(text: str) -> str:
    if text == "":
        return "<empty>"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "J",
        "grad_J_l2",
        "reverse_kl",
        "reward_improvement",
        "reward_vocab_std",
        "reward_softmax_entropy_norm",
        "reward_softmax_kl_uniform",
        "gamma_abs_times_reward_std",
    ]
    out: dict[str, float] = {"length": float(len(rows))}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean()) if values.size else float("nan")
        out[f"{key}_p90"] = float(np.quantile(values, 0.9)) if values.size else float("nan")
        out[f"{key}_max"] = float(values.max()) if values.size else float("nan")
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sequence(path: Path, result: SequenceResult) -> None:
    rows = result.rows
    if not rows:
        return
    x = np.asarray([row["step"] for row in rows])
    token_labels = [row["token_text"] for row in rows]
    max_ticks = 16
    tick_idx = np.linspace(0, len(x) - 1, num=min(max_ticks, len(x)), dtype=int)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"Sequence {result.sequence_id}: token-level misalignment diagnostics", fontsize=14)

    axes[0].plot(x, [row["J"] for row in rows], label="J", color="#1f77b4")
    axes[0].plot(x, [row["grad_J_l2"] for row in rows], label="||grad J||2", color="#d62728", alpha=0.85)
    axes[0].set_ylabel("J / grad")
    axes[0].legend(loc="upper right")

    axes[1].plot(x, [row["reward_softmax_entropy_norm"] for row in rows], label="H(softmax R) / log V", color="#2ca02c")
    axes[1].plot(x, [row["reward_softmax_kl_uniform"] for row in rows], label="KL(softmax R || uniform)", color="#9467bd")
    axes[1].set_ylabel("reward shape")
    axes[1].legend(loc="upper right")

    axes[2].plot(x, [row["reward_vocab_std"] for row in rows], label="std(R)", color="#ff7f0e")
    axes[2].plot(x, [row["gamma_abs_times_reward_std"] for row in rows], label="|gamma| * std(R)", color="#8c564b")
    axes[2].set_ylabel("reward scale")
    axes[2].legend(loc="upper right")

    axes[3].plot(x, [row["reverse_kl"] for row in rows], label="KL(policy || ref)", color="#17becf")
    axes[3].plot(x, [row["reward_improvement"] for row in rows], label="E_policy[R] - E_ref[R]", color="#bcbd22")
    axes[3].set_ylabel("shift")
    axes[3].set_xlabel("completion token step")
    axes[3].legend(loc="upper right")
    axes[3].set_xticks(tick_idx)
    axes[3].set_xticklabels([token_labels[i][:18] for i in tick_idx], rotation=45, ha="right")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_aggregate(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    entropy = np.asarray([row["reward_softmax_entropy_norm"] for row in rows])
    kl_uniform = np.asarray([row["reward_softmax_kl_uniform"] for row in rows])
    reward_std = np.asarray([row["reward_vocab_std"] for row in rows])
    J = np.asarray([row["J"] for row in rows])
    grad = np.asarray([row["grad_J_l2"] for row in rows])
    reverse_kl = np.asarray([row["reverse_kl"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    sc0 = axes[0].scatter(entropy, J, c=reward_std, s=18, cmap="viridis", alpha=0.8)
    axes[0].set_xlabel("H(softmax R) / log V")
    axes[0].set_ylabel("J")
    axes[0].set_title("High reward entropy can still carry J")
    fig.colorbar(sc0, ax=axes[0], label="std(R)")

    sc1 = axes[1].scatter(kl_uniform, J, c=reverse_kl, s=18, cmap="magma", alpha=0.8)
    axes[1].set_xlabel("KL(softmax R || uniform)")
    axes[1].set_ylabel("J")
    axes[1].set_title("Reward non-uniformity vs J")
    fig.colorbar(sc1, ax=axes[1], label="KL(policy || ref)")

    sc2 = axes[2].scatter(reward_std, grad, c=J, s=18, cmap="plasma", alpha=0.8)
    axes[2].set_xlabel("std(R)")
    axes[2].set_ylabel("||grad J||2")
    axes[2].set_title("Reward scale vs gradient")
    fig.colorbar(sc2, ax=axes[2], label="J")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "aggregate_scatter.png", dpi=170)
    plt.close(fig)

    order = np.argsort(entropy)
    sorted_rows = [rows[i] for i in order]
    bins = np.array_split(np.arange(len(sorted_rows)), min(10, len(sorted_rows)))
    bin_x = []
    bin_stats: dict[str, list[float]] = {"J": [], "grad_J_l2": [], "reverse_kl": [], "reward_vocab_std": []}
    sorted_entropy = entropy[order]
    for b in bins:
        if len(b) == 0:
            continue
        bin_x.append(float(sorted_entropy[b].mean()))
        for key in bin_stats:
            values = np.asarray([sorted_rows[int(i)][key] for i in b], dtype=np.float64)
            bin_stats[key].append(float(values.mean()))

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(bin_x, bin_stats["J"], marker="o", label="mean J", color="#1f77b4")
    ax1.plot(bin_x, bin_stats["grad_J_l2"], marker="o", label="mean ||grad J||2", color="#d62728")
    ax1.set_xlabel("reward entropy bin: H(softmax R) / log V")
    ax1.set_ylabel("J / grad")
    ax2 = ax1.twinx()
    ax2.plot(bin_x, bin_stats["reverse_kl"], marker="s", label="mean KL(policy || ref)", color="#17becf")
    ax2.plot(bin_x, bin_stats["reward_vocab_std"], marker="s", label="mean std(R)", color="#ff7f0e")
    ax2.set_ylabel("KL / std(R)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "entropy_binned_trends.png", dpi=170)
    plt.close(fig)


def write_report(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    results: list[SequenceResult],
    all_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if all_rows:
        arrays = {key: np.asarray([row[key] for row in all_rows], dtype=np.float64) for key in numeric_keys(all_rows)}
    else:
        arrays = {}

    def mean(key: str) -> float:
        arr = arrays.get(key)
        return float(arr.mean()) if arr is not None and arr.size else float("nan")

    def quantile(key: str, q: float) -> float:
        arr = arrays.get(key)
        return float(np.quantile(arr, q)) if arr is not None and arr.size else float("nan")

    high_entropy_rows = [
        row
        for row in all_rows
        if row["reward_softmax_entropy_norm"] >= args.high_entropy_threshold
        and row["reward_softmax_kl_uniform"] <= args.low_reward_kl_threshold
    ]
    low_scale_rows = [row for row in all_rows if row["reward_vocab_std"] <= args.low_reward_std_threshold]
    high_j_rows = sorted(all_rows, key=lambda row: row["J"], reverse=True)[:10]

    lines = [
        "# Sequence Misalignment Diagnostic Report",
        "",
        "## Run Configuration",
        "",
        f"- policy model: `{args.policy_model}`",
        f"- reference model: `{args.reference_model}`",
        f"- winner reward model: `{args.winner_model}`",
        f"- loser reward model: `{args.loser_model}`",
        f"- dataset: `{args.dataset_name}:{args.dataset_split}`",
        f"- sampled prompts: `{len(results)}`",
        f"- max generated tokens: `{args.max_new_tokens}`",
        f"- max analyzed tokens per sequence: `{args.max_analysis_tokens}`",
        f"- seed: `{args.seed}`",
        "",
        "## Aggregate Metrics",
        "",
        "| metric | mean | p10 | p50 | p90 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in (
        "J",
        "grad_J_l2",
        "reverse_kl",
        "reward_improvement",
        "reward_vocab_std",
        "reward_softmax_entropy_norm",
        "reward_softmax_kl_uniform",
        "gamma_abs_times_reward_std",
    ):
        lines.append(
            f"| `{key}` | {mean(key):.4g} | {quantile(key, 0.1):.4g} | "
            f"{quantile(key, 0.5):.4g} | {quantile(key, 0.9):.4g} | {quantile(key, 1.0):.4g} |"
        )

    lines.extend(
        [
            "",
            "## Reward-Shape Buckets",
            "",
            (
                f"- high entropy / near-uniform reward rows: `{len(high_entropy_rows)}` of `{len(all_rows)}` "
                f"using entropy >= `{args.high_entropy_threshold}` and KL-to-uniform <= "
                f"`{args.low_reward_kl_threshold}`"
            ),
            f"- low reward-std rows: `{len(low_scale_rows)}` of `{len(all_rows)}` using std(R) <= `{args.low_reward_std_threshold}`",
            "",
            "These buckets are meant as diagnostics, not hard truth labels. A full-vocab softmax can have very high normalized entropy even when its top tail matters, because the Qwen vocabulary is large.",
            "",
            "## Highest-J Token Rows",
            "",
            "| seq | step | token | J | grad_l2 | KL(policy||ref) | std(R) | H_R/logV | KL_R_uniform | gamma*std |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in high_j_rows:
        lines.append(
            f"| {row['sequence_id']} | {row['step']} | `{row['token_text']}` | {row['J']:.4g} | "
            f"{row['grad_J_l2']:.4g} | {row['reverse_kl']:.4g} | {row['reward_vocab_std']:.4g} | "
            f"{row['reward_softmax_entropy_norm']:.4g} | {row['reward_softmax_kl_uniform']:.4g} | "
            f"{row['gamma_abs_times_reward_std']:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- `plots/aggregate_scatter.png`: reward entropy/non-uniformity/scale against J and grad J.",
            "- `plots/entropy_binned_trends.png`: binned trends after sorting token rows by reward entropy.",
            "- `plots/sequence_*.png`: per-token traces for individual generated completions.",
            "",
            "## Theoretical Reading Guide",
            "",
            textwrap.dedent(
                """
                The reward vector is `R = log pi_w - log pi_l`. The J metric first centers
                `R`, then finds an exponential tilt of the reference distribution,
                `q_gamma(v) proportional to pi_ref(v) exp(gamma R_center(v))`, whose
                expected reward matches the policy expected reward. It then computes
                `J = KL(pi_policy || q_gamma)`.

                If `R` is exactly constant, `R_center = 0`, gamma is inactive, and
                `q_gamma = pi_ref`. In that limit J collapses to `KL(pi_policy || pi_ref)`.
                That is numerically stable, but semantically it is no longer evidence of
                reward-model misalignment; it is mostly policy/reference drift measured
                in a direction where the reward model provides no meaningful distinction.

                If `softmax(R)` is high-entropy but non-uniform, small or noisy reward
                differences can still produce nonzero gamma and nonzero J. The key
                diagnostic is whether J and `grad_J` track reward-shape quantities
                (`std(R)`, `KL(softmax R || uniform)`, `|gamma|*std(R)`) or mostly track
                policy/reference shift (`KL(policy || ref)`). Rows with high reward
                entropy, low reward KL-to-uniform, and high J should be treated as
                potentially reward-ambiguous rather than as clean evidence that the
                policy is exploiting a meaningful reward direction.
                """
            ).strip(),
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for key in rows[0]:
        if isinstance(rows[0][key], (int, float)) and not isinstance(rows[0][key], bool):
            keys.append(key)
    return keys


def sample_dataset_rows(args: argparse.Namespace) -> list[tuple[int, list[dict[str, str]]]]:
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    count = min(args.num_prompts, len(dataset))
    if args.sample_random:
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(dataset)), count)
    else:
        indices = list(range(args.start_index, min(args.start_index + count, len(dataset))))
    rows = []
    for idx in indices:
        rows.append((idx, normalize_prompt(dataset[idx][args.prompt_column])))
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.policy_model, args.tokenizer_fallback)
    dtype = torch_dtype(args.dtype)

    print("[load] policy", args.policy_model, "->", args.policy_device, flush=True)
    policy = load_causal_lm(args.policy_model, dtype=dtype, device=args.policy_device)
    print("[load] reference", args.reference_model, "->", args.reference_device, flush=True)
    reference = load_causal_lm(args.reference_model, dtype=dtype, device=args.reference_device)
    print("[load] reward engines", flush=True)
    provider = TokenVocabRewardProvider(
        winner_model=args.winner_model,
        loser_model=args.loser_model,
        winner_device=args.reward_winner_device,
        loser_device=args.reward_loser_device,
        compile=not args.no_reward_compile,
        logprob_dtype=args.reward_logprob_dtype,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    results: list[SequenceResult] = []
    all_token_rows: list[dict[str, Any]] = []
    dataset_rows = sample_dataset_rows(args)
    try:
        for sequence_id, (prompt_index, prompt) in enumerate(dataset_rows):
            prompt_ids = prompt_ids_from_messages(tokenizer, prompt, args.max_prompt_length)
            raw_completion = generate_completion(
                policy,
                tokenizer,
                prompt_ids,
                device=args.policy_device,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            completion_ids = TokenVocabRewardProvider._normalize_completion_ids(
                raw_completion,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            completion_ids = completion_ids[: args.max_analysis_tokens]
            if not completion_ids:
                print(f"[skip] sequence {sequence_id}: empty normalized completion", flush=True)
                continue

            print(
                f"[sequence {sequence_id}] prompt_index={prompt_index} tokens={len(completion_ids)}",
                flush=True,
            )
            policy_logits = completion_logits(
                policy,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                device=args.policy_device,
            )
            reference_logits = completion_logits(
                reference,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                device=args.reference_device,
            ).to(args.policy_device)
            reward_output = provider.compute([prompt_ids], [completion_ids], device=torch.device(args.policy_device))
            rewards = reward_output.rewards[0][: len(completion_ids)].to(args.policy_device)

            rows, summary = token_diagnostics(
                tokenizer=tokenizer,
                policy_logits=policy_logits,
                reference_logits=reference_logits,
                rewards=rewards,
                completion_ids=completion_ids,
                beta=args.beta,
                compute_dtype=metric_dtype(args.compute_dtype),
            )
            prompt_text = prompt_to_text(prompt)
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
            result = SequenceResult(
                sequence_id=sequence_id,
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                completion_text=completion_text,
                rows=rows,
                summary=summary,
            )
            results.append(result)
            for row in rows:
                row.update(
                    {
                        "sequence_id": sequence_id,
                        "prompt_index": prompt_index,
                        "prompt_preview": prompt_text[:240],
                    }
                )
                all_token_rows.append(row)
            plot_sequence(output_dir / "plots" / f"sequence_{sequence_id:03d}.png", result)

            torch.cuda.empty_cache()
    finally:
        provider.close()

    write_csv(output_dir / "token_metrics.csv", all_token_rows)
    summary_rows = []
    for result in results:
        row = dict(result.summary)
        row.update(
            {
                "sequence_id": result.sequence_id,
                "prompt_index": result.prompt_index,
                "prompt_preview": result.prompt_text[:240],
                "completion_preview": result.completion_text[:240],
            }
        )
        summary_rows.append(row)
    write_csv(output_dir / "sequence_summaries.csv", summary_rows)
    plot_aggregate(output_dir, all_token_rows)
    write_report(output_dir, args=args, results=results, all_rows=all_token_rows)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[done] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
