from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import set_seed
from transformers import AutoTokenizer

from trl_misalignment.compat import apply_runtime_compatibility_patches
from trl_misalignment import (
    MisalignmentGRPOTrainer,
    MisalignmentOnlineDPOTrainer,
    TokenVocabRewardProvider,
    VocabMisalignmentConfig,
    make_token_vocab_reward_func,
)

apply_runtime_compatibility_patches()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL/DPO training with token-vocab rewards.")
    parser.add_argument("--trainer", default="grpo", choices=["grpo", "online_dpo"])
    parser.add_argument("--model", required=True, help="Policy model, e.g. a 4B HF model or local path.")
    parser.add_argument("--output-dir", default="runs/token-vocab-reward")
    parser.add_argument("--dataset-name", default="trl-lib/ultrafeedback-prompt")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--tokenizer-fallback", default="Qwen/Qwen3-4B-Instruct-2507")

    parser.add_argument("--reward-url", default=None, help="Paired vocab reward server returning pi_w - pi_l.")
    parser.add_argument("--winner-url", default=None, help="Vocab logprob server for pi_w.")
    parser.add_argument("--loser-url", default=None, help="Vocab logprob server for pi_l.")
    parser.add_argument("--reward-timeout", type=float, default=1200.0)

    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-dtype", default="bfloat16")
    parser.add_argument("--max-model-length", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fsdp", default="")
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default="")
    parser.add_argument("--deepspeed", default=None)

    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", default="server", choices=["server", "colocate"])
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-server-timeout", type=float, default=1800.0)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=2)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-to", default="wandb")

    parser.add_argument("--monitor-j-backprop", action="store_true")
    parser.add_argument("--misalignment-loss-coef", type=float, default=0.0)
    parser.add_argument("--monitor-prompt-reduce", default="sum", choices=["sum", "mean"])
    parser.add_argument("--monitor-compute-dtype", default="float64", choices=["float64", "float32", "none"])
    parser.add_argument("--debug-tokenization-jsonl", default=None)
    parser.add_argument("--debug-tokenization-samples", type=int, default=0)
    return parser.parse_args()


def dataset_with_prompt_column(args: argparse.Namespace):
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    if args.prompt_column != "prompt":
        dataset = dataset.rename_column(args.prompt_column, "prompt")
    dataset = dataset.map(_normalize_prompt_row, desc="Normalizing prompts")
    return dataset


def _normalize_prompt_row(example: dict[str, Any]) -> dict[str, Any]:
    prompt = example["prompt"]
    if isinstance(prompt, str):
        return {"prompt": [{"role": "user", "content": prompt}]}
    if isinstance(prompt, list):
        return {"prompt": prompt}
    raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")


def compute_dtype(name: str) -> Any:
    if name == "none":
        return None
    import torch

    return getattr(torch, name)


def supported_config_kwargs(config_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    fields = getattr(config_cls, "__dataclass_fields__", {})
    if not fields:
        return kwargs
    dropped = sorted(set(kwargs) - set(fields))
    if dropped:
        print(f"[config] Dropping unsupported {config_cls.__name__} kwargs for installed TRL: {dropped}")
    return {key: value for key, value in kwargs.items() if key in fields}


def distributed_training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if args.fsdp:
        kwargs["fsdp"] = args.fsdp
        fsdp_config: dict[str, Any] = {}
        if args.fsdp_transformer_layer_cls_to_wrap:
            fsdp_config["transformer_layer_cls_to_wrap"] = [
                item.strip()
                for item in args.fsdp_transformer_layer_cls_to_wrap.split(",")
                if item.strip()
            ]
        if fsdp_config:
            kwargs["fsdp_config"] = fsdp_config
    if args.deepspeed:
        kwargs["deepspeed"] = args.deepspeed
    return kwargs


def capped_model_length(args: argparse.Namespace) -> int:
    requested = args.max_prompt_length + args.max_completion_length
    if args.max_model_length is None:
        return requested
    return min(requested, args.max_model_length)


def load_tokenizer(model: str, fallback: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", truncation_side="left")
    except Exception as exc:
        print(f"[tokenizer] Failed to load {model!r}: {exc}. Falling back to {fallback!r}.")
        tokenizer = AutoTokenizer.from_pretrained(fallback, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def verify_chat_template_eos_behavior(tokenizer) -> None:
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.eos_token_id is None:
        return

    prompt = [{"role": "user", "content": "Return the word ok."}]
    with_completion = prompt + [{"role": "assistant", "content": "ok"}]
    prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(with_completion, tokenize=True)
    completion_ids = full_ids[len(prompt_ids) :]
    normalized = TokenVocabRewardProvider._normalize_completion_ids(
        completion_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if tokenizer.eos_token_id in completion_ids and (not normalized or normalized[-1] != tokenizer.eos_token_id):
        raise ValueError("Completion token normalization failed to end at EOS.")
    if normalized and completion_ids and len(normalized) != len(completion_ids):
        print(
            "[tokenizer] apply_chat_template leaves trailing token(s) after assistant EOS; "
            "reward scoring will trim after EOS."
        )


def is_main_process_env() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def write_prompt_tokenization_debug(
    *,
    path: str | None,
    sample_count: int,
    dataset: Any,
    tokenizer: Any,
) -> None:
    if not path or sample_count <= 0 or not is_main_process_env():
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    rows = []
    for idx in range(min(sample_count, len(dataset))):
        prompt = dataset[idx]["prompt"]
        prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        row: dict[str, Any] = {
            "source": "dataset_prompt",
            "sample_index": idx,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "pad_equals_eos": pad_token_id is not None and pad_token_id == eos_token_id,
            "prompt_length": len(prompt_ids),
            "prompt_contains_eos": eos_token_id in prompt_ids if eos_token_id is not None else False,
            "prompt_eos_count": prompt_ids.count(eos_token_id) if eos_token_id is not None else 0,
            "prompt_tail_ids": prompt_ids[-32:],
            "prompt_tail_text": tokenizer.decode(prompt_ids[-96:], skip_special_tokens=False),
        }
        rows.append(row)

    with out_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.report_to == "wandb" and "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "trl-misalignment-grpo-online-dpo"

    tokenizer = load_tokenizer(args.model, args.tokenizer_fallback)
    verify_chat_template_eos_behavior(tokenizer)

    dataset = dataset_with_prompt_column(args)
    write_prompt_tokenization_debug(
        path=args.debug_tokenization_jsonl,
        sample_count=args.debug_tokenization_samples,
        dataset=dataset,
        tokenizer=tokenizer,
    )
    provider = TokenVocabRewardProvider(
        reward_url=args.reward_url,
        winner_url=args.winner_url,
        loser_url=args.loser_url,
        timeout=args.reward_timeout,
        request_format="npz",
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    reward_func = make_token_vocab_reward_func(provider, tokenizer)

    backprop_j = args.monitor_j_backprop or args.misalignment_loss_coef != 0.0
    misalignment_config = VocabMisalignmentConfig(
        enabled=True,
        beta=args.beta,
        compute_dtype=compute_dtype(args.monitor_compute_dtype),
        backprop_j=backprop_j,
        j_loss_coef=args.misalignment_loss_coef,
        prompt_reduce=args.monitor_prompt_reduce,
        debug_tokenization_path=args.debug_tokenization_jsonl,
        debug_tokenization_samples=args.debug_tokenization_samples,
    )

    common_kwargs = dict(
        model=args.model,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        processing_class=tokenizer,
        misalignment_config=misalignment_config,
        vocab_reward_provider=provider,
    )
    dist_kwargs = distributed_training_kwargs(args)
    effective_max_model_length = capped_model_length(args)

    if args.trainer == "grpo":
        from trl import GRPOConfig

        training_args = GRPOConfig(
            **supported_config_kwargs(
                GRPOConfig,
                dict(
                    output_dir=args.output_dir,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    warmup_ratio=args.warmup_ratio,
                    lr_scheduler_type=args.lr_scheduler_type,
                    max_grad_norm=args.max_grad_norm,
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    num_train_epochs=args.num_train_epochs,
                    max_steps=args.max_steps,
                    max_prompt_length=args.max_prompt_length,
                    max_completion_length=args.max_completion_length,
                    vllm_max_model_length=effective_max_model_length,
                    num_generations=args.num_generations,
                    beta=args.beta,
                    seed=args.seed,
                    data_seed=args.seed,
                    logging_strategy="steps",
                    logging_steps=args.logging_steps,
                    save_steps=args.save_steps,
                    save_total_limit=args.save_total_limit,
                    save_only_model=True,
                    run_name=args.run_name,
                    bf16=args.bf16,
                    tf32=args.tf32,
                    use_vllm=args.use_vllm,
                    vllm_mode=args.vllm_mode,
                    vllm_server_host=args.vllm_server_host,
                    vllm_server_port=args.vllm_server_port,
                    vllm_server_timeout=args.vllm_server_timeout,
                    vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
                    model_init_kwargs={"dtype": args.model_dtype},
                    report_to=[args.report_to] if args.report_to else [],
                    **dist_kwargs,
                ),
            )
        )
        trainer = MisalignmentGRPOTrainer(args=training_args, **common_kwargs)
    else:
        from trl.experimental.online_dpo import OnlineDPOConfig

        training_args = OnlineDPOConfig(
            **supported_config_kwargs(
                OnlineDPOConfig,
                dict(
                    output_dir=args.output_dir,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    warmup_ratio=args.warmup_ratio,
                    lr_scheduler_type=args.lr_scheduler_type,
                    max_grad_norm=args.max_grad_norm,
                    per_device_train_batch_size=args.per_device_train_batch_size,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    num_train_epochs=args.num_train_epochs,
                    max_steps=args.max_steps,
                    max_length=effective_max_model_length,
                    max_new_tokens=args.max_completion_length,
                    beta=args.beta,
                    seed=args.seed,
                    data_seed=args.seed,
                    logging_strategy="steps",
                    logging_steps=args.logging_steps,
                    save_steps=args.save_steps,
                    save_total_limit=args.save_total_limit,
                    save_only_model=True,
                    run_name=args.run_name,
                    bf16=args.bf16,
                    tf32=args.tf32,
                    dataloader_drop_last=True,
                    use_vllm=args.use_vllm,
                    vllm_mode=args.vllm_mode,
                    vllm_server_host=args.vllm_server_host,
                    vllm_server_port=args.vllm_server_port,
                    vllm_server_timeout=args.vllm_server_timeout,
                    vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
                    model_init_kwargs={"dtype": args.model_dtype, "device_map": None},
                    report_to=[args.report_to] if args.report_to else [],
                    **dist_kwargs,
                ),
            )
        )
        trainer = MisalignmentOnlineDPOTrainer(args=training_args, **common_kwargs)

    try:
        trainer.train()
        trainer.save_state()
    finally:
        provider.close()


if __name__ == "__main__":
    main()
