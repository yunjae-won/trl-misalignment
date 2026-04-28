from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final trainer_state metrics into a CSV.")
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def flatten_last_metrics(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in log_history:
        for key, value in row.items():
            if key in {"epoch", "step"}:
                out[key] = value
            elif isinstance(value, (int, float)):
                out[key] = value
    return out


def main() -> None:
    args = parse_args()
    local_root = Path(args.local_root)
    manifest_rows = load_manifest(Path(args.manifest)) if args.manifest else {}
    rows: list[dict[str, Any]] = []

    for state_path in sorted(local_root.glob("*/trainer_state.json")):
        run_dir = state_path.parent
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        row = flatten_last_metrics(state.get("log_history", []))
        row["run_name"] = run_dir.name
        row["output_dir"] = str(run_dir)
        row.update(manifest_rows.get(run_dir.name, {}))
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            run_name = row.get("run_name")
            if run_name:
                rows[run_name] = row
    return rows


if __name__ == "__main__":
    main()
