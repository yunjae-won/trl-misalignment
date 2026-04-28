from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass
class VocabRewardOutput:
    """Vocab reward for a batch of prompt/completion pairs."""

    rewards: List[torch.Tensor]
    winner_logprobs: Optional[List[torch.Tensor]] = None
    loser_logprobs: Optional[List[torch.Tensor]] = None

    def sequence_rewards(self, completion_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        values = []
        for reward, ids in zip(self.rewards, completion_ids):
            if reward.numel() == 0:
                values.append(torch.tensor(0.0, dtype=torch.float32))
                continue
            token_ids = torch.as_tensor(ids[: reward.shape[0]], dtype=torch.long, device=reward.device)
            values.append(reward.gather(1, token_ids[:, None]).squeeze(1).sum().float().cpu())
        return torch.stack(values)


class TokenVocabRewardProvider:
    """Compute `R_t(v) = log pi_w(v | prefix_t) - log pi_l(v | prefix_t)`.

    The provider supports two deployment modes:

    - HTTP mode: pass `winner_url` and `loser_url` pointing at two
      logprob-engine servers started with vocab-level logprob support.
    - Local mode: pass `winner_model` and `loser_model`; this instantiates two
      `LogprobEngine` objects in the current process with `logprob_level="vocab"`.

    Results are cached per exact `(prompt_ids, output_ids)` tuple to avoid
    recomputing the same reward tensors when a trainer reward function and a
    trainer monitor both request the current rollout.
    """

    def __init__(
        self,
        *,
        reward_url: Optional[str] = None,
        winner_url: Optional[str] = None,
        loser_url: Optional[str] = None,
        winner_model: Optional[str] = None,
        loser_model: Optional[str] = None,
        dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
        winner_device: Optional[str] = None,
        loser_device: Optional[str] = None,
        compile: bool = True,
        request_format: str = "npz",
        timeout: float = 600.0,
        logprob_dtype: str = "float32",
        cache_max_batches: int = 1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        normalize_completions: bool = True,
    ) -> None:
        self.request_format = request_format
        self.cache_max_batches = cache_max_batches
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.normalize_completions = normalize_completions
        self.last_timings: Dict[str, float] = {}
        self._last_scoring_timings: Dict[str, float] = {}
        self._cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], torch.Tensor] = {}
        self._cache_order: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

        if reward_url is not None and (winner_url is not None or loser_url is not None):
            raise ValueError("Use either reward_url or winner/loser URLs, not both.")
        if reward_url is not None and (winner_model is not None or loser_model is not None):
            raise ValueError("Use either reward_url or local winner/loser models, not both.")
        if (winner_url is None) != (loser_url is None):
            raise ValueError("Pass both winner_url and loser_url, or neither.")
        if (winner_model is None) != (loser_model is None):
            raise ValueError("Pass both winner_model and loser_model, or neither.")
        if reward_url is None and winner_url is None and winner_model is None:
            raise ValueError("Configure reward_url, HTTP winner/loser URLs, or local model paths.")
        if winner_url is not None and winner_model is not None:
            raise ValueError("Use either HTTP URLs or local model paths, not both.")

        if reward_url is not None:
            from logprob_engine import LogprobClient

            self._reward = LogprobClient(reward_url, timeout=timeout)
            self._mode = "reward_http"
        elif winner_url is not None:
            from logprob_engine import LogprobClient

            self._winner = LogprobClient(winner_url, timeout=timeout)
            self._loser = LogprobClient(loser_url, timeout=timeout)  # type: ignore[arg-type]
            self._mode = "http"
        else:
            from logprob_engine import LogprobEngine

            self._winner = LogprobEngine(
                winner_model,  # type: ignore[arg-type]
                dtype=dtype,
                attn_implementation=attn_implementation,
                device=winner_device,
                compile=compile,
                logprob_level="vocab",
                logprob_dtype=logprob_dtype,
            )
            self._loser = LogprobEngine(
                loser_model,  # type: ignore[arg-type]
                dtype=dtype,
                attn_implementation=attn_implementation,
                device=loser_device,
                compile=compile,
                logprob_level="vocab",
                logprob_dtype=logprob_dtype,
            )
            self._mode = "local"

    def close(self) -> None:
        clients = (getattr(self, "_reward", None), getattr(self, "_winner", None), getattr(self, "_loser", None))
        for client in clients:
            if client is None:
                continue
            close = getattr(client, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _to_id_lists(x: Any, mask: Optional[Any] = None) -> List[List[int]]:
        if isinstance(x, torch.Tensor):
            x_cpu = x.detach().cpu()
            if mask is not None:
                mask_cpu = mask.detach().cpu().bool() if isinstance(mask, torch.Tensor) else torch.as_tensor(mask).bool()
                return [row[m].tolist() for row, m in zip(x_cpu, mask_cpu)]
            return [row.tolist() for row in x_cpu]
        if mask is not None:
            mask_t = mask.detach().cpu().bool() if isinstance(mask, torch.Tensor) else torch.as_tensor(mask).bool()
            return [[tok for tok, keep in zip(row, m.tolist()) if keep] for row, m in zip(x, mask_t)]
        return [list(row) for row in x]

    @staticmethod
    def _normalize_completion_ids(
        ids: Sequence[int],
        *,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
    ) -> List[int]:
        """Trim completion-only IDs for reward-model scoring.

        `ids` must not include prompt tokens. Prompt-side EOS tokens are valid
        context and should not affect completion trimming.
        """

        out = [int(tok) for tok in ids]
        if pad_token_id is not None and pad_token_id != eos_token_id:
            pad = int(pad_token_id)
            while out and out[-1] == pad:
                out.pop()
        if eos_token_id is not None:
            eos = int(eos_token_id)
            if eos in out:
                last_eos = len(out) - 1 - out[::-1].index(eos)
                out = out[: last_eos + 1]
        return out

    def _process(self, engine_or_client: Any, items: List[Mapping[str, List[int]]]) -> List[torch.Tensor]:
        if self._mode in {"http", "reward_http"}:
            if hasattr(engine_or_client, "logprob_arrays") and self.request_format != "json":
                arrays = engine_or_client.logprob_arrays(items, format=self.request_format)
            else:
                arrays = engine_or_client.logprobs(items, format=self.request_format)
        elif hasattr(engine_or_client, "process_arrays"):
            arrays = engine_or_client.process_arrays(items)
        else:
            arrays = engine_or_client.process(items)
        return [torch.as_tensor(arr, dtype=torch.float32) for arr in arrays]

    def _timed_process(self, engine_or_client: Any, items: List[Mapping[str, List[int]]]) -> Tuple[List[torch.Tensor], float]:
        start = time.perf_counter()
        return self._process(engine_or_client, items), time.perf_counter() - start

    def _remember(
        self,
        key: Tuple[Tuple[int, ...], Tuple[int, ...]],
        value: torch.Tensor,
    ) -> None:
        self._cache[key] = value
        self._cache_order.append(key)
        while len(self._cache_order) > self.cache_max_batches:
            stale = self._cache_order.pop(0)
            self._cache.pop(stale, None)

    def compute(
        self,
        prompt_ids: Any,
        completion_ids: Any,
        *,
        prompt_mask: Optional[Any] = None,
        completion_mask: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> VocabRewardOutput:
        total_start = time.perf_counter()
        prompts = self._to_id_lists(prompt_ids, prompt_mask)
        completions = self._to_id_lists(completion_ids, completion_mask)
        if self.normalize_completions:
            completions = [
                self._normalize_completion_ids(
                    completion,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                )
                for completion in completions
            ]

        items: List[Mapping[str, List[int]]] = []
        keys: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        missing: List[int] = []

        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            if not prompt or not completion:
                raise ValueError("logprob-engine requires non-empty prompt_ids and output_ids.")
            key = (tuple(int(x) for x in prompt), tuple(int(x) for x in completion))
            keys.append(key)
            if key not in self._cache:
                items.append({"prompt_ids": list(key[0]), "output_ids": list(key[1])})
                missing.append(idx)

        winner_seconds = 0.0
        loser_seconds = 0.0
        reward_seconds = 0.0
        combine_start = time.perf_counter()
        if items:
            if self._mode == "reward_http":
                reward, reward_seconds = self._timed_process(self._reward, items)
                for idx, r in zip(missing, reward):
                    self._remember(keys[idx], r.cpu())
            elif self._mode == "http":
                with ThreadPoolExecutor(max_workers=2) as pool:
                    winner_future = pool.submit(self._timed_process, self._winner, items)
                    loser_future = pool.submit(self._timed_process, self._loser, items)
                    winner, winner_seconds = winner_future.result()
                    loser, loser_seconds = loser_future.result()
                for idx, w, l in zip(missing, winner, loser):
                    reward = w - l
                    self._remember(keys[idx], reward.cpu())
            else:
                winner, winner_seconds = self._timed_process(self._winner, items)
                loser, loser_seconds = self._timed_process(self._loser, items)
                for idx, w, l in zip(missing, winner, loser):
                    reward = w - l
                    self._remember(keys[idx], reward.cpu())
        combine_seconds = time.perf_counter() - combine_start

        rewards: List[torch.Tensor] = []
        for key in keys:
            reward = self._cache[key]
            if device is not None:
                reward = reward.to(device)
            rewards.append(reward)

        missing_set = set(missing)
        missing_tokens = sum(len(item["output_ids"]) for item in items)
        cached_tokens = sum(len(keys[idx][1]) for idx in range(len(keys)) if idx not in missing_set)
        timings = {
            "wall_seconds": time.perf_counter() - total_start,
            "reward_seconds": reward_seconds,
            "winner_seconds": winner_seconds,
            "loser_seconds": loser_seconds,
            "score_and_cache_seconds": combine_seconds,
            "missing_items": float(len(items)),
            "cached_items": float(len(keys) - len(items)),
            "missing_tokens": float(missing_tokens),
            "cached_tokens": float(cached_tokens),
        }
        if items:
            self._last_scoring_timings = dict(timings)
        scoring = self._last_scoring_timings if self._last_scoring_timings else timings
        timings.update(
            {
                "scoring_wall_seconds": float(scoring.get("wall_seconds", 0.0)),
                "scoring_reward_seconds": float(scoring.get("reward_seconds", 0.0)),
                "scoring_winner_seconds": float(scoring.get("winner_seconds", 0.0)),
                "scoring_loser_seconds": float(scoring.get("loser_seconds", 0.0)),
                "scoring_score_and_cache_seconds": float(scoring.get("score_and_cache_seconds", 0.0)),
                "scoring_missing_items": float(scoring.get("missing_items", 0.0)),
                "scoring_missing_tokens": float(scoring.get("missing_tokens", 0.0)),
            }
        )
        self.last_timings = timings

        return VocabRewardOutput(
            rewards=rewards,
        )


def pad_vocab_reward(
    rewards: Sequence[torch.Tensor],
    *,
    pad_to: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a ragged list of `[time, vocab]` tensors to `[batch, time, vocab]`."""

    if not rewards:
        raise ValueError("Expected at least one reward tensor.")
    vocab = rewards[0].shape[-1]
    max_len = pad_to if pad_to is not None else max(r.shape[0] for r in rewards)
    out = rewards[0].new_zeros((len(rewards), max_len, vocab))
    mask = torch.zeros((len(rewards), max_len), dtype=torch.long, device=out.device)
    for i, reward in enumerate(rewards):
        if reward.shape[-1] != vocab:
            raise ValueError("All reward tensors must have the same vocab dimension.")
        n = min(reward.shape[0], max_len)
        out[i, :n] = reward[:n].to(out.device)
        mask[i, :n] = 1
    if device is not None:
        out = out.to(device)
        mask = mask.to(device)
    return out, mask


def make_token_vocab_reward_func(
    provider: TokenVocabRewardProvider,
    processing_class: Any,
    *,
    prompt_add_special_tokens: bool = False,
) -> Any:
    """Build a TRL reward function that sums selected-token vocab rewards.

    This is intentionally small: it lets PPO/GRPO/Online-DPO consume the same
    `log pi_w - log pi_l` signal as a scalar sequence reward while trainer
    subclasses can separately monitor the full vocab-level reward tensor.
    """

    tokenizer = getattr(processing_class, "tokenizer", processing_class)

    def _tokenize_prompt(prompt: Any) -> List[int]:
        if isinstance(prompt, str):
            return tokenizer.encode(prompt, add_special_tokens=prompt_add_special_tokens)
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        raise TypeError("Conversational prompts require a tokenizer with apply_chat_template.")

    def reward_func(prompts: Sequence[Any], completion_ids: Sequence[Sequence[int]], **_: Any) -> List[float]:
        prompt_ids = [_tokenize_prompt(prompt) for prompt in prompts]
        output = provider.compute(prompt_ids, completion_ids)
        return output.sequence_rewards(completion_ids).tolist()

    reward_func.__name__ = "token_vocab_reward"
    return reward_func
