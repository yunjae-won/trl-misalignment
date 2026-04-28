from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trl_misalignment.compat import apply_runtime_compatibility_patches


def main() -> None:
    apply_runtime_compatibility_patches()

    from trl.scripts.vllm_serve import main as serve_main
    from trl.scripts.vllm_serve import make_parser

    parser = make_parser(prog="python scripts/trl_vllm_serve_compat.py")
    (script_args,) = parser.parse_args_and_config()
    serve_main(script_args)


if __name__ == "__main__":
    main()
