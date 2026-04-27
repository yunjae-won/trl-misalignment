from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass
class VocabRewardOutput:
    """Vocab reward for a batch of prompt/completion pairs."""

    rewards: List[torch.Tensor]
    winner_logprobs: List[torch.Tensor]
    loser_logprobs: List[torch.Tensor]

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
        cache_max_batches: int = 4,
    ) -> None:
        self.request_format = request_format
        self.cache_max_batches = cache_max_batches
        self._cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._cache_order: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

        if (winner_url is None) != (loser_url is None):
            raise ValueError("Pass both winner_url and loser_url, or neither.")
        if (winner_model is None) != (loser_model is None):
            raise ValueError("Pass both winner_model and loser_model, or neither.")
        if winner_url is None and winner_model is None:
            raise ValueError("Configure either HTTP URLs or local model paths for the reward provider.")
        if winner_url is not None and winner_model is not None:
            raise ValueError("Use either HTTP URLs or local model paths, not both.")

        if winner_url is not None:
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
        for client in (self._winner, self._loser):
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

    def _process(self, engine_or_client: Any, items: List[Mapping[str, List[int]]]) -> List[torch.Tensor]:
        if self._mode == "http":
            arrays = engine_or_client.logprobs(items, format=self.request_format)
        else:
            arrays = engine_or_client.process(items)
        return [torch.as_tensor(arr, dtype=torch.float32) for arr in arrays]

    def _remember(
        self,
        key: Tuple[Tuple[int, ...], Tuple[int, ...]],
        value: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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
        prompts = self._to_id_lists(prompt_ids, prompt_mask)
        completions = self._to_id_lists(completion_ids, completion_mask)

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

        if items:
            winner = self._process(self._winner, items)
            loser = self._process(self._loser, items)
            for idx, w, l in zip(missing, winner, loser):
                reward = w - l
                self._remember(keys[idx], (reward.cpu(), w.cpu(), l.cpu()))

        rewards: List[torch.Tensor] = []
        winner_logprobs: List[torch.Tensor] = []
        loser_logprobs: List[torch.Tensor] = []
        for key in keys:
            reward, winner, loser = self._cache[key]
            if device is not None:
                reward = reward.to(device)
                winner = winner.to(device)
                loser = loser.to(device)
            rewards.append(reward)
            winner_logprobs.append(winner)
            loser_logprobs.append(loser)

        return VocabRewardOutput(
            rewards=rewards,
            winner_logprobs=winner_logprobs,
            loser_logprobs=loser_logprobs,
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
