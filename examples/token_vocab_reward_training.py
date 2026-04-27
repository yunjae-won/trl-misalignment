from __future__ import annotations

import argparse
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

from trl_misalignment import (
    MisalignmentGRPOTrainer,
    MisalignmentOnlineDPOTrainer,
    TokenVocabRewardProvider,
    VocabMisalignmentConfig,
    make_token_vocab_reward_func,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL/DPO training with token-vocab rewards.")
    parser.add_argument("--trainer", default="grpo", choices=["grpo", "online_dpo"])
    parser.add_argument("--model", required=True, help="Policy model, e.g. a 4B HF model or local path.")
    parser.add_argument("--output-dir", default="runs/token-vocab-reward")
    parser.add_argument("--dataset-name", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--dataset-split", default="train_prefs")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--max-train-samples", type=int, default=None)

    parser.add_argument("--winner-url", required=True, help="Vocab logprob server for pi_w.")
    parser.add_argument("--loser-url", required=True, help="Vocab logprob server for pi_l.")
    parser.add_argument("--reward-timeout", type=float, default=1200.0)

    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.04)

    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", default="server", choices=["server", "colocate"])
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--report-to", default="wandb")

    parser.add_argument("--monitor-j-backprop", action="store_true")
    parser.add_argument("--monitor-compute-dtype", default="float64", choices=["float64", "float32", "none"])
    return parser.parse_args()


def dataset_with_prompt_column(args: argparse.Namespace):
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    if args.prompt_column != "prompt":
        dataset = dataset.rename_column(args.prompt_column, "prompt")
    return dataset


def compute_dtype(name: str) -> Any:
    if name == "none":
        return None
    import torch

    return getattr(torch, name)


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left", truncation_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset_with_prompt_column(args)
    provider = TokenVocabRewardProvider(
        winner_url=args.winner_url,
        loser_url=args.loser_url,
        timeout=args.reward_timeout,
        request_format="npz",
    )
    reward_func = make_token_vocab_reward_func(provider, tokenizer)

    misalignment_config = VocabMisalignmentConfig(
        enabled=True,
        beta=args.beta,
        compute_dtype=compute_dtype(args.monitor_compute_dtype),
        backprop_j=args.monitor_j_backprop,
    )

    common_kwargs = dict(
        model=args.model,
        reward_funcs=[reward_func],
        train_dataset=dataset,
        processing_class=tokenizer,
        misalignment_config=misalignment_config,
        vocab_reward_provider=provider,
    )

    if args.trainer == "grpo":
        from trl import GRPOConfig

        training_args = GRPOConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            beta=args.beta,
            bf16=args.bf16,
            use_vllm=args.use_vllm,
            vllm_mode=args.vllm_mode,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            report_to=[args.report_to] if args.report_to else [],
        )
        trainer = MisalignmentGRPOTrainer(args=training_args, **common_kwargs)
    else:
        from trl.experimental.online_dpo import OnlineDPOConfig

        if args.monitor_j_backprop:
            raise ValueError("Online-DPO supports monitoring only; omit --monitor-j-backprop.")
        training_args = OnlineDPOConfig(
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_length=args.max_prompt_length + args.max_completion_length,
            max_new_tokens=args.max_completion_length,
            beta=args.beta,
            bf16=args.bf16,
            use_vllm=args.use_vllm,
            vllm_mode=args.vllm_mode,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            report_to=[args.report_to] if args.report_to else [],
        )
        trainer = MisalignmentOnlineDPOTrainer(args=training_args, **common_kwargs)

    trainer.train()


if __name__ == "__main__":
    main()
