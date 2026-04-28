from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_METRICS = (
    "reward",
    "objective/scores",
    "objective/scores_margin",
    "train_loss",
    "misalignment/J",
    "misalignment/J_per_token",
    "misalignment/J_aux_loss_raw",
    "misalignment/reward_a",
    "misalignment/reward_a_per_token",
    "misalignment/reward_improvement",
    "misalignment/reward_improvement_per_token",
    "misalignment/reward_improvement_over_reverse_kl",
    "misalignment/reverse_kl_divergence",
    "misalignment/reverse_kl_divergence_per_token",
    "misalignment/js_divergence_per_token",
    "misalignment/tv_distance_per_token",
    "misalignment/gamma_bracketed_rate",
    "misalignment/reward_vocab_std_per_token",
    "misalignment/reward_vocab_abs_max_per_token",
    "misalignment/policy_reference_top1_agreement_rate",
    "misalignment/policy_top_is_reward_top_rate",
    "grad_norm",
    "kl",
    "objective/kl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare misalignment ablation results against vanilla baselines.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--analysis-csv", required=True)
    parser.add_argument("--report-md", required=True)
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "misalignment_loss_coef" not in row or row.get("misalignment_loss_coef") == "":
            row["misalignment_loss_coef"] = infer_coef(row.get("run_name", ""))
        if "algo" not in row or row.get("algo") == "":
            row["algo"] = infer_algo(row.get("run_name", ""))
    return rows


def infer_algo(run_name: str) -> str:
    if run_name.startswith("online_dpo"):
        return "online_dpo"
    if run_name.startswith("grpo"):
        return "grpo"
    return "unknown"


def infer_coef(run_name: str) -> str:
    marker = "_jcoef_"
    if marker not in run_name:
        return ""
    value = run_name.split(marker, 1)[1].split("_seed", 1)[0]
    return value.replace("p", ".")


def build_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_algo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_algo[str(row.get("algo", "unknown"))].append(row)

    out: list[dict[str, Any]] = []
    for algo, algo_rows in sorted(by_algo.items()):
        baseline = None
        for row in algo_rows:
            if as_float(row.get("misalignment_loss_coef")) == 0.0:
                baseline = row
                break
        for row in sorted(algo_rows, key=lambda r: as_float(r.get("misalignment_loss_coef")) or 0.0):
            result: dict[str, Any] = {
                "algo": algo,
                "run_name": row.get("run_name", ""),
                "misalignment_loss_coef": row.get("misalignment_loss_coef", ""),
                "step": row.get("step", ""),
            }
            for key in KEY_METRICS:
                value = as_float(row.get(key))
                if value is None:
                    continue
                result[key] = value
                if baseline is not None and row is not baseline:
                    base_value = as_float(baseline.get(key))
                    if base_value is not None:
                        result[f"delta/{key}"] = value - base_value
                        if abs(base_value) > 1e-12:
                            result[f"rel_delta/{key}"] = (value - base_value) / abs(base_value)
            out.append(result)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Misalignment Ablation Analysis",
        "",
        "This report is generated from final `trainer_state.json` metrics. Interpret it as a run-level summary; wandb remains the source for time-series curves.",
        "",
    ]
    by_algo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_algo[str(row.get("algo", "unknown"))].append(row)

    for algo, algo_rows in sorted(by_algo.items()):
        lines.append(f"## {algo}")
        lines.append("")
        headers = [
            "coef",
            "reward/scores",
            "J/token",
            "dJ/token",
            "reward_impr/token",
            "KL/token",
            "gamma_ok",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in sorted(algo_rows, key=lambda r: as_float(r.get("misalignment_loss_coef")) or 0.0):
            reward = row.get("reward", row.get("objective/scores", ""))
            if reward == "":
                reward = row.get("objective/scores", "")
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt(row.get("misalignment_loss_coef")),
                        fmt(reward),
                        fmt(row.get("misalignment/J_per_token")),
                        fmt(row.get("delta/misalignment/J_per_token")),
                        fmt(row.get("misalignment/reward_improvement_per_token")),
                        fmt(row.get("misalignment/reverse_kl_divergence_per_token")),
                        fmt(row.get("misalignment/gamma_bracketed_rate")),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Lower `J/token` is the primary sign that the auxiliary loss reduces vocab-level reward misalignment.",
            "- `reward/scores` tracks the algorithm's scalar reward objective, so the useful region is lower `J/token` without collapsing reward.",
            "- `gamma_ok` near 1 means the gamma solve bracketed reliably; low values make J comparisons less trustworthy.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if abs(number) >= 1000 or (0 < abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.6g}"


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.summary))
    analysis = build_analysis(rows)
    write_csv(analysis, Path(args.analysis_csv))
    write_report(analysis, Path(args.report_md))


if __name__ == "__main__":
    main()
