from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve vocab-level logprobs with logprob-engine.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default=None)
    parser.add_argument("--logprob-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    from logprob_engine import LogprobEngine, create_app

    engine = LogprobEngine(
        args.model,
        dtype=args.dtype,
        attn_implementation=args.attn,
        device=args.device,
        compile=not args.no_compile,
        logprob_level="vocab",
        logprob_dtype=args.logprob_dtype,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
