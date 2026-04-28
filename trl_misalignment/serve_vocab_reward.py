from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

import torch
import uvicorn

from .compat import apply_runtime_compatibility_patches


class PairedVocabRewardEngine:
    """Serve full-vocab reward tensors, `log p_w - log p_l`, from two LMs.

    Returning the reward difference directly avoids sending winner and loser
    full-vocab arrays through HTTP only to subtract them in the trainer process.
    """

    def __init__(
        self,
        *,
        winner_model: str,
        loser_model: str,
        winner_device: str | None,
        loser_device: str | None,
        dtype: str,
        attn_implementation: str | None,
        compile: bool,
        logprob_dtype: str,
        output_dtype: str,
        concurrent: bool,
    ) -> None:
        from logprob_engine import LogprobEngine

        self.winner = LogprobEngine(
            winner_model,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device=winner_device,
            compile=compile,
            logprob_level="vocab",
            logprob_dtype=logprob_dtype,
        )
        self.loser = LogprobEngine(
            loser_model,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device=loser_device,
            compile=compile,
            logprob_level="vocab",
            logprob_dtype=logprob_dtype,
        )
        self.model_name_or_path = f"{winner_model} minus {loser_model}"
        self.torch_dtype = self.winner.torch_dtype
        self.device = f"winner:{self.winner.device},loser:{self.loser.device}"
        self.output_dtype = torch.float16 if output_dtype == "float16" else torch.float32
        self.concurrent = concurrent

    @property
    def vocab_size(self) -> int:
        return self.winner.vocab_size

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.winner.tokenize(text, add_special_tokens=add_special_tokens)

    def _score_pair(self, items: Sequence[Mapping[str, list[int]]]):
        if self.concurrent:
            with ThreadPoolExecutor(max_workers=2) as pool:
                winner_future = pool.submit(self.winner.process_tensors, items)
                loser_future = pool.submit(self.loser.process_tensors, items)
                winner = winner_future.result()
                loser = loser_future.result()
        else:
            winner = self.winner.process_tensors(items)
            loser = self.loser.process_tensors(items)
        return winner, loser

    @torch.inference_mode()
    def process_tensors(self, items: Sequence[Mapping[str, list[int]]]) -> list[torch.Tensor]:
        winner, loser = self._score_pair(items)
        rewards: list[torch.Tensor] = []
        for w, l in zip(winner, loser):
            reward = w.to(dtype=torch.float32)
            reward = reward - l.to(device=reward.device, dtype=torch.float32, non_blocking=True)
            rewards.append(reward)
        return rewards

    @torch.inference_mode()
    def process_arrays(self, items: Sequence[Mapping[str, list[int]]]):
        return [reward.to("cpu", dtype=self.output_dtype).numpy() for reward in self.process_tensors(items)]

    @torch.inference_mode()
    def process(self, items: Sequence[Mapping[str, list[int]]]) -> list[list[float]]:
        return [arr.astype("float32", copy=False).tolist() for arr in self.process_arrays(items)]


def main() -> None:
    apply_runtime_compatibility_patches()
    parser = argparse.ArgumentParser(description="Serve paired vocab rewards with logprob-engine.")
    parser.add_argument("--winner-model", required=True)
    parser.add_argument("--loser-model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--winner-device", default="cuda:0")
    parser.add_argument("--loser-device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn", default=None)
    parser.add_argument("--logprob-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--output-dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-concurrent", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    from logprob_engine import create_app

    engine = PairedVocabRewardEngine(
        winner_model=args.winner_model,
        loser_model=args.loser_model,
        winner_device=args.winner_device,
        loser_device=args.loser_device,
        dtype=args.dtype,
        attn_implementation=args.attn,
        compile=not args.no_compile,
        logprob_dtype=args.logprob_dtype,
        output_dtype=args.output_dtype,
        concurrent=not args.no_concurrent,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
