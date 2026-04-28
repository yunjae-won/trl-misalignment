from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from trl_misalignment.compat import apply_runtime_compatibility_patches
from trl_misalignment.metrics import VocabMisalignmentConfig, batched_vocablevel_misalignment
from trl_misalignment.rewards import TokenVocabRewardProvider

apply_runtime_compatibility_patches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tokenwise vocab-reward misalignment on one unseen prompt/completion sequence."
    )
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--reference-model", default="yunjae-won/ubq30i_qwen4b_sft_both")
    parser.add_argument("--winner-model", default="yunjae-won/ubq30i_qwen4b_sft_yw")
    parser.add_argument("--loser-model", default="yunjae-won/ubq30i_qwen4b_sft_yl")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tokenizer-fallback", default="Qwen/Qwen3-4B-Instruct-2507")

    parser.add_argument("--dataset-name", default="trl-lib/ultrafeedback-prompt")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--sample-index", type=int, default=5000)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--completion-text", default=None)
    parser.add_argument("--completion-ids-json", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-positions", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260428)

    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--reference-device", default="cuda:0")
    parser.add_argument("--winner-device", default="cuda:0")
    parser.add_argument("--loser-device", default="cuda:0")
    parser.add_argument("--metrics-device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--reward-logprob-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--compile-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attn", default=None)

    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--compute-dtype", default="float32", choices=["float32", "float64", "none"])
    parser.add_argument("--noise-std", type=float, default=1e-3)
    parser.add_argument("--noise-samples", type=int, default=8)
    parser.add_argument("--weak-std-quantile", type=float, default=0.25)
    parser.add_argument("--output-dir", default="/yj_data/trl_misalignment/token_sequence_analysis")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def metric_dtype(name: str) -> torch.dtype | None:
    if name == "none":
        return None
    return getattr(torch, name)


def load_tokenizer(model: str, fallback: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", truncation_side="left")
    except Exception as exc:
        print(f"[tokenizer] Failed to load {model!r}: {exc}. Falling back to {fallback!r}.")
        tokenizer = AutoTokenizer.from_pretrained(fallback, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_prompt(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return prompt
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def load_prompt(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.prompt_text is not None:
        return [{"role": "user", "content": args.prompt_text}]
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} outside dataset length {len(dataset)}")
    return normalize_prompt(dataset[args.sample_index][args.prompt_column])


def chat_prompt_ids(tokenizer: Any, prompt: Sequence[Mapping[str, str]]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(list(prompt), add_generation_prompt=True, tokenize=True)
    text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt)
    return tokenizer.encode(text, add_special_tokens=True)


def completion_ids_from_text(tokenizer: Any, prompt_ids: Sequence[int], text: str) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        # Tokenize as an assistant message and strip the prompt prefix so EOS
        # handling matches the chat-template path used by training.
        prompt = [{"role": "user", "content": ""}]
        full = tokenizer.apply_chat_template(
            prompt + [{"role": "assistant", "content": text}],
            tokenize=True,
        )
        empty_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        ids = full[len(empty_prompt) :]
        if ids:
            return ids
    return tokenizer.encode(text, add_special_tokens=False)


def load_model_for_inference(model_name: str, *, device: str, dtype_name: str, attn: str | None = None):
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype(dtype_name),
        "low_cpu_mem_usage": True,
    }
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    return model


def free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def generate_completion_ids(
    *,
    model_name: str,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    args: argparse.Namespace,
) -> list[int]:
    model = load_model_for_inference(model_name, device=args.policy_device, dtype_name=args.dtype, attn=args.attn)
    inputs = torch.tensor([list(prompt_ids)], dtype=torch.long, device=args.policy_device)
    generation = model.generate(
        inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature if args.do_sample else None,
        top_p=args.top_p if args.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    completion = generation[0, inputs.shape[1] :].detach().cpu().tolist()
    del model, inputs, generation
    free_cuda()
    return completion


@torch.inference_mode()
def completion_logits(
    *,
    model_name: str,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    device: str,
    dtype_name: str,
    attn: str | None,
) -> torch.Tensor:
    model = load_model_for_inference(model_name, device=device, dtype_name=dtype_name, attn=attn)
    input_ids = torch.tensor([list(prompt_ids) + list(completion_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :][:, -len(completion_ids) :, :].squeeze(0).float().cpu()
    del model, input_ids, attention_mask, output
    free_cuda()
    return logits


@torch.inference_mode()
def reward_model_logprobs(
    *,
    model_name: str,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    device: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    from logprob_engine import LogprobEngine

    engine = LogprobEngine(
        model_name,
        dtype=args.dtype,
        attn_implementation=args.attn,
        device=device,
        compile=args.compile_reward,
        logprob_level="vocab",
        logprob_dtype=args.reward_logprob_dtype,
    )
    arrays = engine.process_arrays([{"prompt_ids": list(prompt_ids), "output_ids": list(completion_ids)}])
    out = torch.as_tensor(arrays[0], dtype=torch.float32)
    del engine
    free_cuda()
    return out


def kl_from_logps(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    p = logp.exp()
    return (p * (logp - logq)).sum(dim=-1)


def reward_model_similarity(log_w: torch.Tensor, log_l: torch.Tensor) -> dict[str, torch.Tensor]:
    p_w = log_w.exp()
    p_l = log_l.exp()
    log_m = torch.logaddexp(log_w, log_l) - math.log(2.0)
    return {
        "winner_loser_reverse_kl": kl_from_logps(log_w, log_l),
        "winner_loser_forward_kl": kl_from_logps(log_l, log_w),
        "winner_loser_js": 0.5 * kl_from_logps(log_w, log_m) + 0.5 * kl_from_logps(log_l, log_m),
        "winner_loser_tv": 0.5 * (p_w - p_l).abs().sum(dim=-1),
        "winner_loser_top1_agreement": (p_w.argmax(dim=-1) == p_l.argmax(dim=-1)).float(),
    }


def noise_sensitivity(
    *,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    reward: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if args.noise_samples <= 0 or args.noise_std <= 0:
        zeros = torch.zeros(reward.shape[0], device=reward.device)
        return {"J_noise_mean": zeros, "J_noise_std": zeros, "J_noise_cv": zeros}
    samples = []
    generator = torch.Generator(device=reward.device)
    generator.manual_seed(args.seed + 17)
    for _ in range(args.noise_samples):
        noise = torch.randn(reward.shape, generator=generator, device=reward.device, dtype=reward.dtype)
        noise = noise - noise.mean(dim=-1, keepdim=True)
        metrics = batched_vocablevel_misalignment(
            policy_logits,
            reference_logits,
            reward + args.noise_std * noise,
            beta=args.beta,
            compute_dtype=metric_dtype(args.compute_dtype),
        )
        samples.append(metrics["J"].float())
    stacked = torch.stack(samples, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    return {
        "J_noise_mean": mean,
        "J_noise_std": std,
        "J_noise_cv": std / mean.abs().clamp_min(1e-12),
    }


def percentile_of_selected(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    selected = values.gather(1, ids[:, None]).squeeze(1)
    return (values <= selected[:, None]).float().mean(dim=-1)


def token_rows(
    *,
    tokenizer: Any,
    completion_ids: Sequence[int],
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    winner_logps: torch.Tensor,
    loser_logps: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.metrics_device if torch.cuda.is_available() else "cpu")
    A = policy_logits.to(device)
    B = reference_logits.to(device)
    W = winner_logps.to(device)
    L = loser_logps.to(device)
    R = W - L

    config = VocabMisalignmentConfig(beta=args.beta, compute_dtype=metric_dtype(args.compute_dtype))
    token_metrics = batched_vocablevel_misalignment(
        A,
        B,
        R,
        beta=config.beta,
        compute_dtype=config.compute_dtype,
        gamma_iters=config.gamma_iters,
        bracket_iters=config.bracket_iters,
        reward_tol=config.reward_tol,
        ppo_clip_eps=config.ppo_clip_eps,
    )
    rm_metrics = reward_model_similarity(W, L)
    sens = noise_sensitivity(policy_logits=A, reference_logits=B, reward=R, args=args)

    policy_logps = F.log_softmax(A, dim=-1)
    ref_logps = F.log_softmax(B, dim=-1)
    policy_probs = policy_logps.exp()
    ref_probs = ref_logps.exp()

    ids = torch.tensor(list(completion_ids), dtype=torch.long, device=device)
    selected_reward = R.gather(1, ids[:, None]).squeeze(1)
    selected_policy_prob = policy_probs.gather(1, ids[:, None]).squeeze(1)
    selected_ref_prob = ref_probs.gather(1, ids[:, None]).squeeze(1)
    selected_reward_percentile = percentile_of_selected(R, ids)

    reward_std = token_metrics["reward_vocab_std"].float()
    weak_threshold = torch.quantile(reward_std, float(args.weak_std_quantile)).item()

    rows: list[dict[str, Any]] = []
    for i, token_id in enumerate(completion_ids):
        row: dict[str, Any] = {
            "position": i,
            "token_id": int(token_id),
            "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
            "selected_reward": float(selected_reward[i].detach().cpu()),
            "selected_reward_percentile": float(selected_reward_percentile[i].detach().cpu()),
            "selected_policy_prob": float(selected_policy_prob[i].detach().cpu()),
            "selected_reference_prob": float(selected_ref_prob[i].detach().cpu()),
            "weak_reward_signal": bool(float(reward_std[i].detach().cpu()) <= weak_threshold),
        }
        for key, values in token_metrics.items():
            if values.ndim == 1:
                value = values[i]
                row[key] = bool(value.item()) if value.dtype == torch.bool else float(value.detach().cpu())
        for key, values in rm_metrics.items():
            row[key] = float(values[i].detach().cpu())
        for key, values in sens.items():
            row[key] = float(values[i].detach().cpu())
        rows.append(row)

    summary = summarize_rows(rows, weak_threshold)
    del A, B, W, L, R
    free_cuda()
    return rows, summary


def summarize_rows(rows: Sequence[Mapping[str, Any]], weak_threshold: float) -> dict[str, Any]:
    def values(key: str, subset: Sequence[Mapping[str, Any]] = rows) -> list[float]:
        return [float(row[key]) for row in subset if isinstance(row.get(key), (int, float))]

    def mean(xs: Sequence[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    weak = [row for row in rows if row.get("weak_reward_signal")]
    strong = [row for row in rows if not row.get("weak_reward_signal")]
    summary = {
        "tokens": len(rows),
        "weak_reward_threshold_std": weak_threshold,
        "weak_reward_tokens": len(weak),
        "strong_reward_tokens": len(strong),
        "mean_J": mean(values("J")),
        "mean_J_weak_reward": mean(values("J", weak)),
        "mean_J_strong_reward": mean(values("J", strong)),
        "mean_reward_vocab_std": mean(values("reward_vocab_std")),
        "mean_reward_vocab_std_weak": mean(values("reward_vocab_std", weak)),
        "mean_reward_vocab_std_strong": mean(values("reward_vocab_std", strong)),
        "mean_winner_loser_js": mean(values("winner_loser_js")),
        "mean_policy_entropy": mean(values("entropy_a")),
        "mean_reference_entropy": mean(values("entropy_b")),
        "mean_reverse_kl": mean(values("reverse_kl_divergence")),
        "mean_J_noise_cv": mean(values("J_noise_cv")),
        "max_J_noise_cv": max(values("J_noise_cv")) if values("J_noise_cv") else None,
        "gamma_bracketed_rate": mean([1.0 if row.get("gamma_bracketed") else 0.0 for row in rows]),
    }
    if rows:
        ranked = sorted(rows, key=lambda row: float(row.get("J", 0.0)), reverse=True)[:10]
        summary["top_J_positions"] = [
            {
                "position": row["position"],
                "token_id": row["token_id"],
                "token_text": row["token_text"],
                "J": row.get("J"),
                "policy_entropy": row.get("entropy_a"),
                "reward_vocab_std": row.get("reward_vocab_std"),
                "winner_loser_js": row.get("winner_loser_js"),
            }
            for row in ranked
        ]
    return summary


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric_series(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        out.append(float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else float("nan"))
    return out


def scaled_points(xs: Sequence[float], ys: Sequence[float], box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    x0, y0, width, height = box
    finite_y = [y for y in ys if math.isfinite(y)]
    y_min = min(finite_y) if finite_y else 0.0
    y_max = max(finite_y) if finite_y else 1.0
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.5
        y_max += 0.5
    x_min = min(xs) if xs else 0
    x_max = max(xs) if xs else 1
    if x_max == x_min:
        x_max = x_min + 1
    pts = []
    for x, y in zip(xs, ys):
        if not math.isfinite(y):
            continue
        px = x0 + (x - x_min) / (x_max - x_min) * width
        py = y0 + height - (y - y_min) / (y_max - y_min) * height
        pts.append((px, py))
    return pts


def line_path(points: Sequence[tuple[float, float]]) -> str:
    if not points:
        return ""
    first, *rest = points
    return "M " + f"{first[0]:.2f} {first[1]:.2f} " + " ".join(f"L {x:.2f} {y:.2f}" for x, y in rest)


def svg_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    keys: Sequence[tuple[str, str]],
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    xs = [float(row["position"]) for row in rows]
    pieces = [
        f'<g transform="translate({x},{y})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#d0d7de"/>',
        f'<text x="10" y="18" font-size="13" font-family="monospace" fill="#24292f">{html.escape(title)}</text>',
        f'<line x1="40" y1="{height - 28}" x2="{width - 12}" y2="{height - 28}" stroke="#8c959f"/>',
        f'<line x1="40" y1="28" x2="40" y2="{height - 28}" stroke="#8c959f"/>',
    ]
    box = (40, 28, width - 52, height - 56)
    for key, color in keys:
        ys = numeric_series(rows, key)
        points = scaled_points(xs, ys, box)
        pieces.append(f'<path d="{line_path(points)}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        pieces.append(
            f'<text x="{width - 210}" y="{32 + 16 * keys.index((key, color))}" '
            f'font-size="11" font-family="monospace" fill="{color}">{html.escape(key)}</text>'
        )
    return "\n".join(pieces + ["</g>"])


def scatter_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    y_key: str,
    color_key: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    xs = numeric_series(rows, x_key)
    ys = numeric_series(rows, y_key)
    cs = numeric_series(rows, color_key)
    x0, y0, plot_w, plot_h = (40, 28, width - 52, height - 56)
    finite_x = [v for v in xs if math.isfinite(v)]
    finite_y = [v for v in ys if math.isfinite(v)]
    finite_c = [v for v in cs if math.isfinite(v)]
    xmin, xmax = (min(finite_x), max(finite_x)) if finite_x else (0.0, 1.0)
    ymin, ymax = (min(finite_y), max(finite_y)) if finite_y else (0.0, 1.0)
    cmin, cmax = (min(finite_c), max(finite_c)) if finite_c else (0.0, 1.0)
    if xmax == xmin:
        xmax += 1
    if ymax == ymin:
        ymax += 1
    if cmax == cmin:
        cmax += 1
    pieces = [
        f'<g transform="translate({x},{y})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#d0d7de"/>',
        f'<text x="10" y="18" font-size="13" font-family="monospace" fill="#24292f">'
        f'{html.escape(y_key)} vs {html.escape(x_key)}</text>',
        f'<line x1="40" y1="{height - 28}" x2="{width - 12}" y2="{height - 28}" stroke="#8c959f"/>',
        f'<line x1="40" y1="28" x2="40" y2="{height - 28}" stroke="#8c959f"/>',
        f'<text x="44" y="{height - 8}" font-size="10" font-family="monospace" fill="#57606a">{html.escape(x_key)}</text>',
    ]
    for row, xv, yv, cv in zip(rows, xs, ys, cs):
        if not all(math.isfinite(v) for v in (xv, yv, cv)):
            continue
        px = x0 + (xv - xmin) / (xmax - xmin) * plot_w
        py = y0 + plot_h - (yv - ymin) / (ymax - ymin) * plot_h
        heat = (cv - cmin) / (cmax - cmin)
        red = int(42 + 180 * heat)
        blue = int(180 - 120 * heat)
        color = f"rgb({red},95,{blue})"
        pieces.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}">'
            f'<title>pos {row["position"]}: {html.escape(str(row["token_text"]))}</title></circle>'
        )
    return "\n".join(pieces + ["</g>"])


def write_svg_dashboard(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1200
    height = 950
    panels = [
        svg_panel(rows, title="Misalignment and policy/reference KL", keys=(("J", "#cf222e"), ("reverse_kl_divergence", "#0969da")), x=20, y=30, width=560, height=210),
        svg_panel(rows, title="Reward signal strength", keys=(("reward_vocab_std", "#8250df"), ("winner_loser_js", "#1a7f37")), x=610, y=30, width=560, height=210),
        svg_panel(rows, title="Entropy", keys=(("entropy_a", "#bf8700"), ("entropy_b", "#57606a")), x=20, y=270, width=560, height=210),
        svg_panel(rows, title="Selected-token reward and percentile", keys=(("selected_reward", "#cf222e"), ("selected_reward_percentile", "#0969da")), x=610, y=270, width=560, height=210),
        svg_panel(rows, title="Reward-noise sensitivity", keys=(("J_noise_cv", "#8250df"), ("gamma_star", "#1a7f37")), x=20, y=510, width=560, height=210),
        scatter_panel(rows, x_key="entropy_a", y_key="J", color_key="reward_vocab_std", x=610, y=510, width=560, height=210),
    ]
    body = "\n".join(panels)
    legend = (
        '<text x="20" y="900" font-size="13" font-family="monospace" fill="#24292f">'
        'Weak reward signal = bottom quantile of reward_vocab_std. '
        'Hover points in the scatter for token text/position.</text>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#f6f8fa"/>\n'
        '<text x="20" y="22" font-size="16" font-family="monospace" fill="#24292f">'
        'Tokenwise vocab-reward misalignment dashboard</text>\n'
        f"{body}\n{legend}\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def write_report(
    *,
    path: Path,
    args: argparse.Namespace,
    prompt: Sequence[Mapping[str, str]],
    completion_text: str,
    completion_ids: Sequence[int],
    summary: Mapping[str, Any],
    dashboard_name: str,
) -> None:
    top = summary.get("top_J_positions", [])
    top_lines = "\n".join(
        f"- pos `{row['position']}` token `{row['token_text']!r}`: "
        f"J={row['J']:.4g}, entropy={row['policy_entropy']:.4g}, "
        f"reward_std={row['reward_vocab_std']:.4g}, winner_loser_js={row['winner_loser_js']:.4g}"
        for row in top
    )
    text = f"""# Tokenwise Misalignment Analysis

## Inputs

- policy: `{args.policy_model}`
- reference: `{args.reference_model}`
- winner reward model: `{args.winner_model}`
- loser reward model: `{args.loser_model}`
- sample index: `{args.sample_index}`
- analyzed tokens: `{len(completion_ids)}`
- beta: `{args.beta}`
- reward noise probe: Gaussian std `{args.noise_std}`, samples `{args.noise_samples}`

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```

## Dashboard

![tokenwise dashboard]({dashboard_name})

## Highest-J Positions

{top_lines}

## Prompt

```json
{json.dumps(prompt, indent=2, ensure_ascii=False)}
```

## Completion

```text
{completion_text}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer_name = args.tokenizer or args.reference_model or args.policy_model
    tokenizer = load_tokenizer(tokenizer_name, args.tokenizer_fallback)

    prompt = load_prompt(args)
    prompt_ids = chat_prompt_ids(tokenizer, prompt)

    if args.completion_ids_json:
        completion_ids = json.loads(Path(args.completion_ids_json).read_text(encoding="utf-8"))
    elif args.completion_text is not None:
        completion_ids = completion_ids_from_text(tokenizer, prompt_ids, args.completion_text)
    else:
        completion_ids = generate_completion_ids(
            model_name=args.policy_model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            args=args,
        )

    normalized_completion_ids = TokenVocabRewardProvider._normalize_completion_ids(
        completion_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    trailing_after_eos = len(completion_ids) - len(normalized_completion_ids)
    if args.max_positions and len(normalized_completion_ids) > args.max_positions:
        normalized_completion_ids = normalized_completion_ids[: args.max_positions]
    if not normalized_completion_ids:
        raise ValueError("No completion tokens to analyze after normalization/truncation.")

    run_name = args.run_name or f"sample{args.sample_index}_tokens{len(normalized_completion_ids)}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analysis] prompt tokens={len(prompt_ids)} completion tokens={len(normalized_completion_ids)}")
    print(f"[analysis] trailing tokens removed after completion EOS={trailing_after_eos}")
    print("[analysis] computing policy logits")
    policy_logits = completion_logits(
        model_name=args.policy_model,
        prompt_ids=prompt_ids,
        completion_ids=normalized_completion_ids,
        device=args.policy_device,
        dtype_name=args.dtype,
        attn=args.attn,
    )
    print("[analysis] computing reference logits")
    if args.reference_model == args.policy_model:
        reference_logits = policy_logits.clone()
    else:
        reference_logits = completion_logits(
            model_name=args.reference_model,
            prompt_ids=prompt_ids,
            completion_ids=normalized_completion_ids,
            device=args.reference_device,
            dtype_name=args.dtype,
            attn=args.attn,
        )
    print("[analysis] computing winner reward-model logprobs")
    winner_logps = reward_model_logprobs(
        model_name=args.winner_model,
        prompt_ids=prompt_ids,
        completion_ids=normalized_completion_ids,
        device=args.winner_device,
        args=args,
    )
    print("[analysis] computing loser reward-model logprobs")
    loser_logps = reward_model_logprobs(
        model_name=args.loser_model,
        prompt_ids=prompt_ids,
        completion_ids=normalized_completion_ids,
        device=args.loser_device,
        args=args,
    )

    rows, summary = token_rows(
        tokenizer=tokenizer,
        completion_ids=normalized_completion_ids,
        policy_logits=policy_logits,
        reference_logits=reference_logits,
        winner_logps=winner_logps,
        loser_logps=loser_logps,
        args=args,
    )
    summary.update(
        {
            "prompt_token_count": len(prompt_ids),
            "raw_completion_token_count": len(completion_ids),
            "analyzed_completion_token_count": len(normalized_completion_ids),
            "trailing_tokens_removed_after_eos": trailing_after_eos,
            "policy_model": args.policy_model,
            "reference_model": args.reference_model,
            "winner_model": args.winner_model,
            "loser_model": args.loser_model,
            "sample_index": args.sample_index,
        }
    )

    completion_text = tokenizer.decode(normalized_completion_ids, skip_special_tokens=False)
    (output_dir / "completion_ids.json").write_text(json.dumps(normalized_completion_ids), encoding="utf-8")
    (output_dir / "prompt_completion.txt").write_text(
        "PROMPT\n" + json.dumps(prompt, indent=2, ensure_ascii=False) + "\n\nCOMPLETION\n" + completion_text,
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(rows, output_dir / "per_token_metrics.csv")
    write_svg_dashboard(rows, output_dir / "token_misalignment_dashboard.svg")
    write_report(
        path=output_dir / "report.md",
        args=args,
        prompt=prompt,
        completion_text=completion_text,
        completion_ids=normalized_completion_ids,
        summary=summary,
        dashboard_name="token_misalignment_dashboard.svg",
    )
    print(f"[analysis] wrote {output_dir}")


if __name__ == "__main__":
    main()
