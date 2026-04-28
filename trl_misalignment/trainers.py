from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .compat import apply_runtime_compatibility_patches
from .metrics import VocabMisalignmentConfig, completion_vocab_misalignment, detached_scalar_logs
from .rewards import TokenVocabRewardProvider, pad_vocab_reward

apply_runtime_compatibility_patches()


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
        self._misalignment_capture: Dict[str, Tuple[torch.Tensor, ...]] = {}
        self._tokenization_debug_written = False

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

    def _log_reward_provider_timings(self) -> None:
        if self.vocab_reward_provider is None:
            return
        timings = getattr(self.vocab_reward_provider, "last_timings", None)
        if not timings:
            return
        device = getattr(self.accelerator, "device", torch.device("cpu"))
        for key, value in timings.items():
            self._append_trainer_metric(
                f"reward_engine/{key}",
                torch.tensor(float(value), device=device),
            )

    def _record_tokenization_state(
        self,
        *,
        source: str,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> None:
        eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        prompt_mask_bool = prompt_mask.detach().bool()
        completion_mask_bool = completion_mask.detach().bool()

        prompt_lengths = prompt_mask_bool.sum(dim=1).float()
        completion_lengths = completion_mask_bool.sum(dim=1).float()
        if eos_token_id is None:
            prompt_contains_eos = torch.zeros_like(prompt_mask_bool)
            completion_contains_eos = torch.zeros_like(completion_mask_bool)
        else:
            prompt_contains_eos = (prompt_ids.detach() == eos_token_id).logical_and(prompt_mask_bool)
            completion_contains_eos = (completion_ids.detach() == eos_token_id).logical_and(completion_mask_bool)

        ends_eos = []
        trailing_after_last_eos = []
        normalized_lengths = []
        for ids, mask in zip(completion_ids.detach().cpu(), completion_mask_bool.detach().cpu(), strict=True):
            masked_ids = ids[mask].tolist()
            normalized = TokenVocabRewardProvider._normalize_completion_ids(
                masked_ids,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            normalized_lengths.append(float(len(normalized)))
            ends_eos.append(float(bool(normalized) and eos_token_id is not None and normalized[-1] == eos_token_id))
            if eos_token_id is not None and eos_token_id in masked_ids:
                last_eos = len(masked_ids) - 1 - masked_ids[::-1].index(eos_token_id)
                trailing_after_last_eos.append(float(len(masked_ids) - last_eos - 1))
            else:
                trailing_after_last_eos.append(0.0)

        device = completion_ids.device
        self._append_trainer_metric("tokenization/prompt_length", prompt_lengths.mean())
        self._append_trainer_metric("tokenization/completion_length", completion_lengths.mean())
        self._append_trainer_metric(
            "tokenization/normalized_completion_length",
            torch.tensor(normalized_lengths, device=device).mean(),
        )
        self._append_trainer_metric("tokenization/prompt_contains_eos_rate", prompt_contains_eos.any(dim=1).float().mean())
        self._append_trainer_metric("tokenization/completion_has_eos_rate", completion_contains_eos.any(dim=1).float().mean())
        self._append_trainer_metric("tokenization/completion_ends_eos_rate", torch.tensor(ends_eos, device=device).mean())
        self._append_trainer_metric(
            "tokenization/trailing_tokens_after_last_completion_eos",
            torch.tensor(trailing_after_last_eos, device=device).mean(),
        )

        debug_path = self.misalignment_config.debug_tokenization_path
        debug_samples = int(self.misalignment_config.debug_tokenization_samples or 0)
        if not debug_path or debug_samples <= 0 or self._tokenization_debug_written:
            return
        if not getattr(self.accelerator, "is_main_process", True):
            return

        rows = self._tokenization_debug_rows(
            source=source,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            max_samples=debug_samples,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        path = Path(debug_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        self._tokenization_debug_written = True

    def _tokenization_debug_rows(
        self,
        *,
        source: str,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        max_samples: int,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        tokenizer = self.processing_class
        decode = getattr(tokenizer, "decode", None)
        for idx, (p_ids, p_mask, c_ids, c_mask) in enumerate(
            zip(
                prompt_ids.detach().cpu(),
                prompt_mask.detach().cpu().bool(),
                completion_ids.detach().cpu(),
                completion_mask.detach().cpu().bool(),
                strict=True,
            )
        ):
            if idx >= max_samples:
                break
            prompt_list = [int(tok) for tok in p_ids[p_mask].tolist()]
            completion_list = [int(tok) for tok in c_ids[c_mask].tolist()]
            normalized = TokenVocabRewardProvider._normalize_completion_ids(
                completion_list,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            if eos_token_id is not None and eos_token_id in completion_list:
                last_eos = len(completion_list) - 1 - completion_list[::-1].index(eos_token_id)
                trailing_after_last_eos = len(completion_list) - last_eos - 1
            else:
                trailing_after_last_eos = 0

            row: dict[str, Any] = {
                "source": source,
                "sample_index": idx,
                "global_step": int(getattr(getattr(self, "state", None), "global_step", -1)),
                "eos_token_id": eos_token_id,
                "pad_token_id": pad_token_id,
                "pad_equals_eos": pad_token_id is not None and pad_token_id == eos_token_id,
                "prompt_length": len(prompt_list),
                "completion_length": len(completion_list),
                "normalized_completion_length": len(normalized),
                "prompt_contains_eos": eos_token_id in prompt_list if eos_token_id is not None else False,
                "completion_eos_count": completion_list.count(eos_token_id) if eos_token_id is not None else 0,
                "completion_ends_eos_after_normalization": bool(normalized)
                and eos_token_id is not None
                and normalized[-1] == eos_token_id,
                "trailing_tokens_after_last_completion_eos": trailing_after_last_eos,
                "prompt_tail_ids": prompt_list[-16:],
                "completion_tail_ids": completion_list[-32:],
                "normalized_completion_tail_ids": normalized[-32:],
            }
            if decode is not None:
                row["prompt_tail_text"] = decode(prompt_list[-64:], skip_special_tokens=False)
                row["completion_text"] = decode(completion_list, skip_special_tokens=False)
                row["normalized_completion_text"] = decode(normalized, skip_special_tokens=False)
            rows.append(row)
        return rows


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
        self._record_tokenization_state(
            source="grpo_rollout",
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )

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

        with torch.no_grad():
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
        self._log_reward_provider_timings()
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

    def _scaled_j_loss(self, j_loss: torch.Tensor) -> torch.Tensor:
        coef = float(self.misalignment_config.j_loss_coef)
        return j_loss * coef

    def _compute_grpo_j_aux_loss(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
        if not self.misalignment_config.enabled or not self.misalignment_config.backprop_j:
            return None
        if self.vocab_reward_provider is None:
            raise ValueError("Misalignment J backprop requires `vocab_reward_provider`.")

        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        if "tool_mask" in inputs:
            completion_mask = completion_mask * inputs["tool_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, inputs["completion_mask"]], dim=1)
        logits_to_keep = completion_ids.shape[1]

        forward_kwargs = {
            key: inputs[key]
            for key in (
                "pixel_values",
                "image_grid_thw",
                "pixel_attention_mask",
                "image_sizes",
                "token_type_ids",
                "mm_token_type_ids",
                "image_position_ids",
            )
            if key in inputs
        }

        policy_logits = self._completion_logits_from_model(
            model,
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
        self._log_reward_provider_timings()
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
        return metrics["prompt"].get("J", None).mean()

    def _compute_loss(self, model: torch.nn.Module, inputs: Dict[str, Any]):
        loss = super()._compute_loss(model, inputs)
        raw_j_loss = self._compute_grpo_j_aux_loss(model, inputs)
        if raw_j_loss is not None:
            scaled_j_loss = self._scaled_j_loss(raw_j_loss)
            loss = loss + scaled_j_loss
            self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss_raw", raw_j_loss)
            self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss", scaled_j_loss)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        # In the non-Liger path, GRPOTrainer.compute_loss dispatches to our
        # _compute_loss above, so the aux term has already been applied.
        if getattr(self, "use_liger_kernel", False):
            raw_j_loss = self._compute_grpo_j_aux_loss(model, inputs)
            if raw_j_loss is not None:
                scaled_j_loss = self._scaled_j_loss(raw_j_loss)
                loss = loss + scaled_j_loss
                self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss_raw", raw_j_loss)
                self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss", scaled_j_loss)
        return loss


try:
    from trl.experimental.online_dpo.online_dpo_trainer import OnlineDPOTrainer
except Exception:  # pragma: no cover
    OnlineDPOTrainer = object  # type: ignore[assignment]

try:
    from accelerate.utils import broadcast_object_list, gather_object
    from transformers.training_args import OptimizerNames
    from trl.data_utils import apply_chat_template, is_conversational
    from trl.experimental.utils import empty_cache
except Exception:  # pragma: no cover
    broadcast_object_list = None  # type: ignore[assignment]
    gather_object = None  # type: ignore[assignment]
    OptimizerNames = None  # type: ignore[assignment]
    apply_chat_template = None  # type: ignore[assignment]
    is_conversational = None  # type: ignore[assignment]
    empty_cache = None  # type: ignore[assignment]


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
        super().__init__(*args, **kwargs)

    def _generate_vllm(self, prompts, images=None):
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id

        if self.vllm_mode == "server":
            completion_ids, prompt_ids = self._generate_vllm_server(prompts, images)
        elif self.vllm_mode == "colocate":
            completion_ids, prompt_ids = self._generate_vllm_colocate(prompts, images)
        else:
            raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got {self.vllm_mode!r}.")

        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]

        max_tokens = self.generation_config.max_tokens
        normalized_completion_ids = []
        for ids in completion_ids:
            ids = TokenVocabRewardProvider._normalize_completion_ids(
                ids,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            if not ids:
                ids = [eos_token_id]
            elif ids[-1] != eos_token_id and len(ids) < max_tokens:
                ids = ids + [eos_token_id]
            normalized_completion_ids.append(ids[:max_tokens])

        completion_mask = [
            [1] * len(ids) + [0] * (max_tokens - len(ids))
            for ids in normalized_completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in normalized_completion_ids]

        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _generate_vllm_server(self, prompts, images=None):
        if gather_object is None or broadcast_object_list is None or apply_chat_template is None:
            raise RuntimeError("Online-DPO vLLM server mode requires compatible Accelerate and TRL imports.")

        has_images = images is not None

        if hasattr(self, "_last_loaded_step") and self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        elif not hasattr(self, "_last_loaded_step"):
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        if is_conversational({"prompt": prompts[0]}):
            prompts_text = [apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts]
        else:
            prompts_text = prompts

        all_prompts = gather_object(prompts_text)
        if has_images:
            all_images = gather_object(images)

        if self.accelerator.is_main_process:
            ordered_images = [[img] if img is not None else None for img in all_images] if has_images else None
            flat_completion_ids = self.vllm_client.generate(
                prompts=all_prompts,
                images=ordered_images,
                n=self.num_generations,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=-1 if self.top_k is None else self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.generation_config.max_tokens,
                structured_outputs_regex=self.structured_outputs_regex
                if hasattr(self, "structured_outputs_regex")
                else None,
                generation_kwargs=self.args.generation_kwargs,
            )["completion_ids"]

            total_prompts = len(all_prompts)
            completion_ids = [
                flat_completion_ids[prompt_idx * self.num_generations + generation_idx]
                for generation_idx in range(self.num_generations)
                for prompt_idx in range(total_prompts)
            ]
        else:
            completion_ids = [None] * (len(all_prompts) * self.num_generations)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)

        local_batch = len(prompts)
        local_start = self.accelerator.process_index * local_batch
        total_batch = len(all_prompts)
        local_completion_ids = []
        for generation_idx in range(self.num_generations):
            offset = generation_idx * total_batch + local_start
            local_completion_ids.extend(completion_ids[offset : offset + local_batch])

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        unpadded_prompt_ids = [
            row[mask.bool()].tolist()
            for row, mask in zip(prompt_inputs["input_ids"], prompt_inputs["attention_mask"], strict=True)
        ]
        prompt_ids = []
        for _ in range(self.num_generations):
            prompt_ids.extend(unpadded_prompt_ids)
        return local_completion_ids, prompt_ids

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
            logits_for_metric = logits if role == "policy" and self.misalignment_config.backprop_j else logits.detach()
            self._misalignment_capture[role] = (
                logits_for_metric,
                prompt_ids.detach(),
                prompt_mask.detach(),
                completion_ids.detach(),
                completion_mask.detach(),
            )
        return logprobs

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.misalignment_config.enabled or not self.misalignment_config.backprop_j:
            self._misalignment_capture.clear()
            loss = super().training_step(model, inputs, num_items_in_batch)
            if self.misalignment_config.enabled:
                self._add_online_dpo_misalignment()
            return loss
        return self._training_step_with_j_backprop(model, inputs, num_items_in_batch)

    def _add_online_dpo_misalignment(self) -> Optional[torch.Tensor]:
        if self.vocab_reward_provider is None:
            raise ValueError("Misalignment monitoring requires `vocab_reward_provider`.")
        if "policy" not in self._misalignment_capture or "reference" not in self._misalignment_capture:
            return None
        policy_logits, prompt_ids, prompt_mask, completion_ids, completion_mask = self._misalignment_capture["policy"]
        reference_logits = self._misalignment_capture["reference"][0]
        self._record_tokenization_state(
            source="online_dpo_rollout",
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
        )
        reward_output = self.vocab_reward_provider.compute(
            prompt_ids,
            completion_ids,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            device=completion_ids.device,
        )
        self._log_reward_provider_timings()
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
        if self.misalignment_config.backprop_j and "J" in metrics["prompt"]:
            return metrics["prompt"]["J"].mean()
        return None

    def _training_step_with_j_backprop(self, model, inputs, num_items_in_batch=None):
        if OptimizerNames is None or is_conversational is None or empty_cache is None:
            raise RuntimeError("Online-DPO J backprop requires compatible TRL and Transformers imports.")

        self._misalignment_capture.clear()
        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)

        has_images = "image" in inputs
        images = None
        if has_images:
            images = inputs["image"]
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [{"type": "image"}, {"type": "text", "text": content}]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(prompts, images)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts, images)

        contain_eos_token = torch.any(completion_ids == self.eos_token_id, dim=-1)

        vision_inputs = None
        if has_images and self.is_vision_model and not self.args.use_vllm:
            vision_inputs = {}
            kwargs = {"images": [[img] for img in images]}
            processed = self.processing_class(
                text=[""] * len(images),
                return_tensors="pt",
                **kwargs,
            )
            model_device = getattr(model, "device", None)
            model_dtype = getattr(model, "dtype", None)
            if model_device is None and hasattr(model, "module"):
                model_device = model.module.device
                model_dtype = model.module.dtype
            if "pixel_values" in processed:
                vision_inputs["pixel_values"] = (
                    processed["pixel_values"].to(model_device, dtype=model_dtype).repeat(2, 1, 1, 1)
                )
            if "pixel_attention_mask" in processed:
                vision_inputs["pixel_attention_mask"] = processed["pixel_attention_mask"].to(model_device).repeat(2, 1)
            if "image_sizes" in processed:
                vision_inputs["image_sizes"] = processed["image_sizes"].to(model_device).repeat(2, 1)
            if "image_grid_thw" in processed:
                vision_inputs["image_grid_thw"] = processed["image_grid_thw"].to(model_device).repeat(2, 1)

        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(
                    self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs
                )
            else:
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(
                        self.model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs
                    )

        device = logprobs.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        completion_ids_list = [completion_ids[i].tolist() for i in range(completion_ids.shape[0])]

        reward_kwargs = {}
        keys = [key for key in inputs if key not in ["prompt"]]
        for key in keys:
            if isinstance(inputs[key], (list, tuple)):
                reward_kwargs[key] = inputs[key] * 2
            else:
                reward_kwargs[key] = inputs[key]

        rewards = self._calculate_rewards_from_functions(
            prompts=2 * prompts,
            completions=completions,
            completion_ids_list=completion_ids_list,
            **reward_kwargs,
        )

        if self.args.missing_eos_penalty is not None:
            rewards[~contain_eos_token] -= self.args.missing_eos_penalty

        first_half, second_half = rewards.split(batch_size)
        mask = first_half >= second_half

        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

        loss = losses.mean()

        raw_j_loss = self._add_online_dpo_misalignment()
        if raw_j_loss is not None:
            scaled_j_loss = raw_j_loss * float(self.misalignment_config.j_loss_coef)
            loss = loss + scaled_j_loss
            self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss_raw", raw_j_loss)
            self._append_trainer_metric(f"{self.misalignment_config.log_prefix}/J_aux_loss", scaled_j_loss)

        if self.reward_funcs is not None:
            scores_margin = rewards[chosen_indices] - rewards[rejected_indices]
            self.stats["objective/scores_margin"].append(
                self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
            )
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(rewards.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        if self.reward_funcs is not None:
            rlhf_reward = rewards + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())

        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps


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
