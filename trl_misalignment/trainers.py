from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch

from .metrics import VocabMisalignmentConfig, completion_vocab_misalignment, detached_scalar_logs
from .rewards import TokenVocabRewardProvider, pad_vocab_reward


class VocabMisalignmentMixin:
    """Shared helpers for TRL trainer subclasses."""

    misalignment_config: VocabMisalignmentConfig
    vocab_reward_provider: Optional[TokenVocabRewardProvider]

    def _init_vocab_misalignment(
        self,
        *,
        misalignment_config: Optional[VocabMisalignmentConfig],
        vocab_reward_provider: Optional[TokenVocabRewardProvider],
    ) -> None:
        self.misalignment_config = misalignment_config or VocabMisalignmentConfig(enabled=False)
        self.vocab_reward_provider = vocab_reward_provider
        self._misalignment_capture: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _completion_logits_from_model(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        *,
        temperature: float,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        kwargs = dict(forward_kwargs or {})
        kwargs["attention_mask"] = attention_mask
        kwargs["use_cache"] = False
        output = model(input_ids=input_ids, **kwargs)
        logits = output.logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:, :]
        return logits / temperature

    def _reference_model_context(self):
        if getattr(self, "ref_model", None) is not None:
            return self.ref_model, nullcontext()
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "disable_adapter"):
            return model, model.disable_adapter()
        return model, nullcontext()

    def _append_trainer_metric(self, key: str, value: torch.Tensor) -> None:
        mode = "train" if self.model.training else "eval"
        scalar = value.detach().float()
        if hasattr(self, "_metrics"):
            gathered = self.accelerator.gather(scalar).nanmean().item()
            self._metrics[mode][key].append(gathered)
        elif hasattr(self, "stats"):
            gathered = self.accelerator.gather_for_metrics(scalar).nanmean().item()
            self.stats.setdefault(key, []).append(gathered)

    def _log_prompt_misalignment(self, prompt_metrics: Dict[str, torch.Tensor]) -> None:
        logs = detached_scalar_logs(
            prompt_metrics,
            prefix=self.misalignment_config.log_prefix,
            keys=self.misalignment_config.log_keys,
        )
        for key, value in logs.items():
            self._append_trainer_metric(key, value)


try:
    from trl.trainer.grpo_trainer import GRPOTrainer
except Exception:  # pragma: no cover - import depends on installed TRL version
    GRPOTrainer = object  # type: ignore[assignment]


class MisalignmentGRPOTrainer(VocabMisalignmentMixin, GRPOTrainer):
    """GRPO trainer with vocab-level misalignment monitoring.

    The subclass delegates rollout generation, vLLM synchronization, reward
    shaping, grouping, and GRPO loss construction to upstream TRL. It adds one
    optional extra forward pass over the generated completions to collect
    full-vocab policy/reference logits for the misalignment metric.
    """

    def __init__(
        self,
        *args: Any,
        misalignment_config: Optional[VocabMisalignmentConfig] = None,
        vocab_reward_provider: Optional[TokenVocabRewardProvider] = None,
        **kwargs: Any,
    ) -> None:
        self._init_vocab_misalignment(
            misalignment_config=misalignment_config,
            vocab_reward_provider=vocab_reward_provider,
        )
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs: Any) -> Dict[str, Any]:
        output = super()._generate_and_score_completions(inputs)
        if self.misalignment_config.enabled:
            self._add_grpo_misalignment(output)
        return output

    def _add_grpo_misalignment(self, output: Dict[str, Any]) -> None:
        if self.vocab_reward_provider is None:
            raise ValueError("Misalignment monitoring requires `vocab_reward_provider`.")

        prompt_ids = output["prompt_ids"]
        prompt_mask = output["prompt_mask"]
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        if "tool_mask" in output:
            completion_mask = completion_mask * output["tool_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, output["completion_mask"]], dim=1)
        logits_to_keep = completion_ids.shape[1]

        forward_kwargs = {
            key: output[key]
            for key in (
                "pixel_values",
                "image_grid_thw",
                "pixel_attention_mask",
                "image_sizes",
                "token_type_ids",
                "mm_token_type_ids",
                "image_position_ids",
            )
            if key in output
        }

        policy_context = nullcontext() if self.misalignment_config.backprop_j else torch.no_grad()
        with policy_context:
            policy_logits = self._completion_logits_from_model(
                self.model,
                input_ids,
                attention_mask,
                logits_to_keep,
                temperature=float(getattr(self, "temperature", 1.0)),
                forward_kwargs=forward_kwargs,
            )

        ref_model, ref_context = self._reference_model_context()
        with torch.no_grad(), ref_context:
            reference_logits = self._completion_logits_from_model(
                ref_model,
                input_ids,
                attention_mask,
                logits_to_keep,
                temperature=float(getattr(self, "temperature", 1.0)),
                forward_kwargs=forward_kwargs,
            )

        reward_output = self.vocab_reward_provider.compute(
            prompt_ids,
            completion_ids,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            device=completion_ids.device,
        )
        vocab_rewards, reward_mask = pad_vocab_reward(
            reward_output.rewards,
            pad_to=completion_ids.shape[1],
            device=completion_ids.device,
        )
        metric_mask = completion_mask * reward_mask
        metrics = completion_vocab_misalignment(
            policy_logits,
            reference_logits,
            vocab_rewards,
            metric_mask,
            self.misalignment_config,
        )
        self._log_prompt_misalignment(metrics["prompt"])
        if self.misalignment_config.backprop_j and "J" in metrics["prompt"]:
            # Keep per-sequence shape so TRL's generation-batch splitter can
            # slice this field together with the other rollout tensors.
            output["misalignment_j_loss"] = metrics["prompt"]["J"]

    def _compute_loss(self, model: torch.nn.Module, inputs: Dict[str, Any]):
        loss = super()._compute_loss(model, inputs)
        j_loss = inputs.get("misalignment_j_loss")
        if self.misalignment_config.enabled and self.misalignment_config.backprop_j and j_loss is not None:
            j_loss = j_loss.mean()
            loss = loss + j_loss
            self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss", j_loss)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        # In the non-Liger path, GRPOTrainer.compute_loss dispatches to our
        # _compute_loss above, so the aux term has already been applied.
        if getattr(self, "use_liger_kernel", False):
            j_loss = inputs.get("misalignment_j_loss")
            if self.misalignment_config.enabled and self.misalignment_config.backprop_j and j_loss is not None:
                j_loss = j_loss.mean()
                loss = loss + j_loss
                self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss", j_loss)
        return loss


try:
    from trl.experimental.online_dpo.online_dpo_trainer import OnlineDPOTrainer
except Exception:  # pragma: no cover
    OnlineDPOTrainer = object  # type: ignore[assignment]


class MisalignmentOnlineDPOTrainer(VocabMisalignmentMixin, OnlineDPOTrainer):
    """Online-DPO trainer with post-step vocab-misalignment monitoring."""

    def __init__(
        self,
        *args: Any,
        misalignment_config: Optional[VocabMisalignmentConfig] = None,
        vocab_reward_provider: Optional[TokenVocabRewardProvider] = None,
        **kwargs: Any,
    ) -> None:
        self._init_vocab_misalignment(
            misalignment_config=misalignment_config,
            vocab_reward_provider=vocab_reward_provider,
        )
        if self.misalignment_config.backprop_j:
            raise ValueError("Online-DPO currently supports monitoring only; set backprop_j=False.")
        super().__init__(*args, **kwargs)

    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs=None):
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        model_kwargs = {"attention_mask": prompt_completion_mask}
        if vision_inputs is not None:
            model_kwargs.update(vision_inputs)

        output = model(prompt_completion_ids, **model_kwargs)
        prompt_len = prompt_ids.size(1)
        start_idx = prompt_len - 1 if prompt_len > 0 else 0
        end_idx = -1 if prompt_len > 0 else None
        logits = output.logits[:, start_idx:end_idx] / float(getattr(self, "temperature", 1.0))
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)

        if self.misalignment_config.enabled:
            role = "policy"
            if (getattr(self, "ref_model", None) is not None and model is self.ref_model) or (
                getattr(self, "ref_model", None) is None and "policy" in self._misalignment_capture
            ):
                role = "reference"
            self._misalignment_capture[role] = (
                logits.detach(),
                prompt_ids.detach(),
                completion_ids.detach(),
                completion_mask.detach(),
            )
        return logprobs

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._misalignment_capture.clear()
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.misalignment_config.enabled:
            self._add_online_dpo_misalignment()
        return loss

    def _add_online_dpo_misalignment(self) -> None:
        if self.vocab_reward_provider is None:
            raise ValueError("Misalignment monitoring requires `vocab_reward_provider`.")
        if "policy" not in self._misalignment_capture or "reference" not in self._misalignment_capture:
            return
        policy_logits, prompt_ids, completion_ids, completion_mask = self._misalignment_capture["policy"]
        reference_logits = self._misalignment_capture["reference"][0]
        reward_output = self.vocab_reward_provider.compute(
            prompt_ids,
            completion_ids,
            completion_mask=completion_mask,
            device=completion_ids.device,
        )
        vocab_rewards, reward_mask = pad_vocab_reward(
            reward_output.rewards,
            pad_to=completion_ids.shape[1],
            device=completion_ids.device,
        )
        metrics = completion_vocab_misalignment(
            policy_logits,
            reference_logits,
            vocab_rewards,
            completion_mask * reward_mask,
            self.misalignment_config,
        )
        self._log_prompt_misalignment(metrics["prompt"])


try:
    from trl.experimental.ppo.ppo_trainer import PPOTrainer
except Exception:  # pragma: no cover
    PPOTrainer = object  # type: ignore[assignment]


class MisalignmentPPOTrainer(VocabMisalignmentMixin, PPOTrainer):
    """PPO extension point for token-vocab reward experiments.

    Upstream TRL's experimental PPO trainer currently keeps rollout,
    reward-model scoring, PPO minibatching, and logging inside one `train`
    method. This subclass keeps the public constructor compatible and stores
    the shared config/provider so research code can use the same reward
    provider. For loss-time `J` backprop with PPO, copy the upstream PPO
    `train` method and call `completion_vocab_misalignment` beside the existing
    `logits/ref_logits` tensors before they are deleted.
    """

    def __init__(
        self,
        *args: Any,
        misalignment_config: Optional[VocabMisalignmentConfig] = None,
        vocab_reward_provider: Optional[TokenVocabRewardProvider] = None,
        **kwargs: Any,
    ) -> None:
        self._init_vocab_misalignment(
            misalignment_config=misalignment_config,
            vocab_reward_provider=vocab_reward_provider,
        )
        if self.misalignment_config.enabled:
            raise NotImplementedError(
                "MisalignmentPPOTrainer is scaffolded but PPO monitoring/backprop needs an adapted PPO train loop. "
                "Use MisalignmentGRPOTrainer for the full implementation, or set enabled=False."
            )
        super().__init__(*args, **kwargs)
