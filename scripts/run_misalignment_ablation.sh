#!/usr/bin/env bash
set -euo pipefail

# Research run plan:
# - GPUs 0-3: policy training with accelerate.
# - GPUs 4-5: TRL vLLM generation server, restarted for every run so weights reset.
# - GPU 6: winner vocab reward model server.
# - GPU 7: loser vocab reward model server.
# - Model checkpoints stay local under LOCAL_ROOT.
# - Persistent metadata, service logs, and summary CSVs go under /yj_data; no model checkpoints are written there.
# - Primary metric history is logged to a new wandb project/run group.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

POLICY_MODEL="${POLICY_MODEL:-yunjae-won/ubq30i_qwen4b_sft_both}"
WINNER_MODEL="${WINNER_MODEL:-yunjae-won/ubq30i_qwen4b_sft_yw}"
LOSER_MODEL="${LOSER_MODEL:-yunjae-won/ubq30i_qwen4b_sft_yl}"
DATASET_NAME="${DATASET_NAME:-trl-lib/ultrafeedback-prompt}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"

SEED="${SEED:-20260428}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-seed${SEED}}"
WANDB_PROJECT="${WANDB_PROJECT:-trl-misalignment-grpo-online-dpo-20260428}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_ID}}"

LOCAL_ROOT="${LOCAL_ROOT:-/root/trl-misalignment/runs/misalignment_ablation/${RUN_ID}}"
PERSIST_ROOT="${PERSIST_ROOT:-/yj_data/trl_misalignment/${RUN_ID}}"
LOG_ROOT="${PERSIST_ROOT}/logs"

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
VLLM_GPUS="${VLLM_GPUS:-4,5}"
REWARD_WINNER_GPU="${REWARD_WINNER_GPU:-6}"
REWARD_LOSER_GPU="${REWARD_LOSER_GPU:-7}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
WINNER_PORT="${WINNER_PORT:-8101}"
LOSER_PORT="${LOSER_PORT:-8102}"
VLLM_TP="${VLLM_TP:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
REWARD_COMPILE="${REWARD_COMPILE:-1}"
REWARD_LOGPROB_DTYPE="${REWARD_LOGPROB_DTYPE:-float32}"
REWARD_OUTPUT_DTYPE="${REWARD_OUTPUT_DTYPE:-float32}"
REWARD_WARMUP="${REWARD_WARMUP:-1}"
REWARD_SERVER_MODE="${REWARD_SERVER_MODE:-paired}"
REWARD_SERVER_CONCURRENT="${REWARD_SERVER_CONCURRENT:-1}"

MAX_STEPS="${MAX_STEPS:-200}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
REWARD_WARMUP_TOKENS="${REWARD_WARMUP_TOKENS:-${MAX_COMPLETION_LENGTH}}"
REWARD_WARMUP_ITEMS="${REWARD_WARMUP_ITEMS:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant_with_warmup}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
BETA="${BETA:-0.04}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
DEBUG_TOKENIZATION_SAMPLES="${DEBUG_TOKENIZATION_SAMPLES:-4}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
FSDP="${FSDP:-}"
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
REWARD_SHARE_GPU="${REWARD_SHARE_GPU:-0}"
TRAIN_NUM_PROCESSES="${TRAIN_NUM_PROCESSES:-4}"

ALGOS="${ALGOS:-grpo online_dpo}"
GRPO_COEFS="${GRPO_COEFS:-0 0.001 0.003}"
ONLINE_DPO_COEFS="${ONLINE_DPO_COEFS:-0 0.001 0.003}"

ACCELERATE_PORT_BASE="${ACCELERATE_PORT_BASE:-29510}"

mkdir -p "${LOCAL_ROOT}" "${LOG_ROOT}"

export POLICY_MODEL WINNER_MODEL LOSER_MODEL DATASET_NAME DATASET_SPLIT
export PERSIST_ROOT
export SEED MAX_STEPS MAX_TRAIN_SAMPLES MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_GENERATIONS LEARNING_RATE BETA
export LR_SCHEDULER_TYPE WARMUP_RATIO MAX_GRAD_NORM MAX_MODEL_LEN
export DEBUG_TOKENIZATION_SAMPLES
export GRADIENT_CHECKPOINTING FSDP FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP DEEPSPEED_CONFIG REWARD_SHARE_GPU
export REWARD_COMPILE REWARD_LOGPROB_DTYPE REWARD_OUTPUT_DTYPE REWARD_WARMUP REWARD_WARMUP_TOKENS REWARD_WARMUP_ITEMS
export REWARD_SERVER_MODE REWARD_SERVER_CONCURRENT
export WANDB_PROJECT
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRL_EXPERIMENTAL_SILENCE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

WINNER_PID=""
LOSER_PID=""
VLLM_PID=""

cleanup() {
  stop_process_group "${VLLM_PID}"
  stop_process_group "${WINNER_PID}"
  stop_process_group "${LOSER_PID}"
}
trap cleanup EXIT

stop_process_group() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    sleep 2
    kill -- "-${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

wait_health() {
  local base_url="$1"
  local name="$2"
  local attempts="${3:-900}"
  local pid="${4:-}"
  local log_file="${5:-}"
  for _ in $(seq 1 "${attempts}"); do
    if curl -fsS "${base_url}/health" >/dev/null 2>&1 || curl -fsS "${base_url}/health/" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} process exited while waiting for ${base_url}" >&2
      return 1
    fi
    if [[ -n "${log_file}" ]] && [[ -f "${log_file}" ]] && grep -qE 'Traceback|RuntimeError|ValueError|WorkerProc failed|Engine core initialization failed' "${log_file}"; then
      echo "${name} logged a startup failure while waiting for ${base_url}" >&2
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "Timed out waiting for ${name} at ${base_url}" >&2
  return 1
}

warmup_reward_server() {
  local base_url="$1"
  local name="$2"
  python - "${base_url}" "${REWARD_WARMUP_TOKENS}" "${REWARD_WARMUP_ITEMS}" "${name}" <<'PY'
import sys
import time

from logprob_engine import LogprobClient

base_url, token_count, item_count, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
client = LogprobClient(base_url, timeout=1800)
items = [
    {
        "prompt_ids": [1, 2, 3, 4 + i],
        "output_ids": list(range(5, 5 + token_count)),
    }
    for i in range(item_count)
]
start = time.perf_counter()
arrays = client.logprob_arrays(items, format="npz")
elapsed = time.perf_counter() - start
print(
    f"{name} warmup_seconds={elapsed:.3f} items={item_count} "
    f"shape={arrays[0].shape} dtype={arrays[0].dtype}",
    flush=True,
)
client.close()
PY
}

warmup_reward_servers() {
  if [[ "${REWARD_WARMUP}" == "0" ]]; then
    return 0
  fi
  if [[ "${REWARD_SERVER_MODE}" == "paired" ]]; then
    warmup_reward_server "http://${VLLM_HOST}:${WINNER_PORT}" "paired_reward" >>"${LOG_ROOT}/reward_warmup.log" 2>&1
    return 0
  fi
  if [[ "${REWARD_SHARE_GPU}" == "1" || "${REWARD_WINNER_GPU}" == "${REWARD_LOSER_GPU}" ]]; then
    warmup_reward_server "http://${VLLM_HOST}:${WINNER_PORT}" "winner" >>"${LOG_ROOT}/reward_warmup.log" 2>&1
    warmup_reward_server "http://${VLLM_HOST}:${LOSER_PORT}" "loser" >>"${LOG_ROOT}/reward_warmup.log" 2>&1
    return 0
  fi
  warmup_reward_server "http://${VLLM_HOST}:${WINNER_PORT}" "winner" >>"${LOG_ROOT}/reward_warmup.log" 2>&1 &
  local winner_warmup_pid="$!"
  warmup_reward_server "http://${VLLM_HOST}:${LOSER_PORT}" "loser" >>"${LOG_ROOT}/reward_warmup.log" 2>&1 &
  local loser_warmup_pid="$!"
  wait "${winner_warmup_pid}"
  wait "${loser_warmup_pid}"
}

start_reward_servers() {
  local reward_compile_args=()
  local reward_concurrency_args=()
  local loser_visible_gpu="${REWARD_LOSER_GPU}"
  if [[ "${REWARD_COMPILE}" == "0" ]]; then
    reward_compile_args+=(--no-compile)
  fi
  if [[ "${REWARD_SERVER_CONCURRENT}" == "0" ]]; then
    reward_concurrency_args+=(--no-concurrent)
  fi
  if [[ "${REWARD_SHARE_GPU}" == "1" ]]; then
    loser_visible_gpu="${REWARD_WINNER_GPU}"
    reward_concurrency_args+=(--no-concurrent)
  fi

  if [[ "${REWARD_SERVER_MODE}" == "paired" ]]; then
    local reward_visible_gpus="${REWARD_WINNER_GPU},${REWARD_LOSER_GPU}"
    local loser_device="cuda:1"
    if [[ "${REWARD_SHARE_GPU}" == "1" || "${REWARD_WINNER_GPU}" == "${REWARD_LOSER_GPU}" ]]; then
      reward_visible_gpus="${REWARD_WINNER_GPU}"
      loser_device="cuda:0"
    fi

    CUDA_VISIBLE_DEVICES="${reward_visible_gpus}" setsid python -m trl_misalignment.serve_vocab_reward \
      --winner-model "${WINNER_MODEL}" \
      --loser-model "${LOSER_MODEL}" \
      --winner-device cuda:0 \
      --loser-device "${loser_device}" \
      --host "${VLLM_HOST}" \
      --port "${WINNER_PORT}" \
      --dtype bfloat16 \
      --logprob-dtype "${REWARD_LOGPROB_DTYPE}" \
      --output-dtype "${REWARD_OUTPUT_DTYPE}" \
      "${reward_compile_args[@]}" \
      "${reward_concurrency_args[@]}" \
      >"${LOG_ROOT}/reward_paired.log" 2>&1 &
    WINNER_PID="$!"
    LOSER_PID=""

    wait_health "http://${VLLM_HOST}:${WINNER_PORT}" "paired reward server" 900 "${WINNER_PID}" "${LOG_ROOT}/reward_paired.log"
    warmup_reward_servers
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${REWARD_WINNER_GPU}" setsid python -m trl_misalignment.serve_vocab_logprobs \
    --model "${WINNER_MODEL}" \
    --device cuda:0 \
    --host "${VLLM_HOST}" \
    --port "${WINNER_PORT}" \
    --dtype bfloat16 \
    --logprob-dtype "${REWARD_LOGPROB_DTYPE}" \
    "${reward_compile_args[@]}" \
    >"${LOG_ROOT}/reward_winner.log" 2>&1 &
  WINNER_PID="$!"

  CUDA_VISIBLE_DEVICES="${loser_visible_gpu}" setsid python -m trl_misalignment.serve_vocab_logprobs \
    --model "${LOSER_MODEL}" \
    --device cuda:0 \
    --host "${VLLM_HOST}" \
    --port "${LOSER_PORT}" \
    --dtype bfloat16 \
    --logprob-dtype "${REWARD_LOGPROB_DTYPE}" \
    "${reward_compile_args[@]}" \
    >"${LOG_ROOT}/reward_loser.log" 2>&1 &
  LOSER_PID="$!"

  wait_health "http://${VLLM_HOST}:${WINNER_PORT}" "winner reward server" 900 "${WINNER_PID}" "${LOG_ROOT}/reward_winner.log"
  wait_health "http://${VLLM_HOST}:${LOSER_PORT}" "loser reward server" 900 "${LOSER_PID}" "${LOG_ROOT}/reward_loser.log"
  warmup_reward_servers
}

start_vllm_server() {
  local run_name="$1"
  local vllm_log="${LOG_ROOT}/vllm_${run_name}.log"
  CUDA_VISIBLE_DEVICES="${VLLM_GPUS}" setsid python scripts/trl_vllm_serve_compat.py \
    --model "${POLICY_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor_parallel_size "${VLLM_TP}" \
    --dtype bfloat16 \
    --max_model_len "${MAX_MODEL_LEN}" \
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    >"${vllm_log}" 2>&1 &
  VLLM_PID="$!"
  wait_health "http://${VLLM_HOST}:${VLLM_PORT}" "vLLM server" 900 "${VLLM_PID}" "${vllm_log}"
}

stop_vllm_server() {
  stop_process_group "${VLLM_PID}"
  VLLM_PID=""
}

write_manifest_row() {
  local run_name="$1"
  local algo="$2"
  local coef="$3"
  local output_dir="$4"
  python - "$PERSIST_ROOT/manifest.jsonl" "$run_name" "$algo" "$coef" "$output_dir" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest, run_name, algo, coef, output_dir = sys.argv[1:]
row = {
    "run_name": run_name,
    "algo": algo,
    "misalignment_loss_coef": float(coef),
    "output_dir": output_dir,
    "tokenization_debug_jsonl": str(Path(os.environ["PERSIST_ROOT"]) / "tokenization_debug" / f"{run_name}.jsonl"),
    "seed": int(os.environ["SEED"]),
    "policy_model": os.environ["POLICY_MODEL"],
    "winner_model": os.environ["WINNER_MODEL"],
    "loser_model": os.environ["LOSER_MODEL"],
    "dataset_name": os.environ["DATASET_NAME"],
    "dataset_split": os.environ["DATASET_SPLIT"],
    "max_steps": int(os.environ["MAX_STEPS"]),
    "max_train_samples": int(os.environ["MAX_TRAIN_SAMPLES"]),
    "max_prompt_length": int(os.environ["MAX_PROMPT_LENGTH"]),
    "max_completion_length": int(os.environ["MAX_COMPLETION_LENGTH"]),
    "per_device_train_batch_size": int(os.environ["PER_DEVICE_TRAIN_BATCH_SIZE"]),
    "gradient_accumulation_steps": int(os.environ["GRADIENT_ACCUMULATION_STEPS"]),
    "num_generations": int(os.environ["NUM_GENERATIONS"]),
    "learning_rate": float(os.environ["LEARNING_RATE"]),
    "lr_scheduler_type": os.environ["LR_SCHEDULER_TYPE"],
    "warmup_ratio": float(os.environ["WARMUP_RATIO"]),
    "max_grad_norm": float(os.environ["MAX_GRAD_NORM"]),
    "beta": float(os.environ["BETA"]),
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "gradient_checkpointing": os.environ["GRADIENT_CHECKPOINTING"] != "0",
    "fsdp": os.environ["FSDP"],
    "fsdp_transformer_layer_cls_to_wrap": os.environ["FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP"],
    "deepspeed_config": os.environ["DEEPSPEED_CONFIG"],
    "reward_server_mode": os.environ["REWARD_SERVER_MODE"],
    "reward_server_concurrent": os.environ["REWARD_SERVER_CONCURRENT"] != "0",
    "reward_share_gpu": os.environ["REWARD_SHARE_GPU"] != "0",
    "debug_tokenization_samples": int(os.environ["DEBUG_TOKENIZATION_SAMPLES"]),
    "reward_compile": os.environ["REWARD_COMPILE"] != "0",
    "reward_logprob_dtype": os.environ["REWARD_LOGPROB_DTYPE"],
    "reward_output_dtype": os.environ["REWARD_OUTPUT_DTYPE"],
    "reward_warmup": os.environ["REWARD_WARMUP"] != "0",
    "reward_warmup_tokens": int(os.environ["REWARD_WARMUP_TOKENS"]),
    "reward_warmup_items": int(os.environ["REWARD_WARMUP_ITEMS"]),
}
with Path(manifest).open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

run_training() {
  local algo="$1"
  local coef="$2"
  local run_index="$3"
  local coef_label="${coef//./p}"
  local run_name="${algo}_jcoef_${coef_label}_seed${SEED}"
  local output_dir="${LOCAL_ROOT}/${run_name}"
  local train_log="${LOG_ROOT}/train_${run_name}.log"
  local token_debug="${PERSIST_ROOT}/tokenization_debug/${run_name}.jsonl"
  local accelerate_port=$((ACCELERATE_PORT_BASE + run_index))
  local distributed_args=()
  local reward_args=()

  if [[ "${GRADIENT_CHECKPOINTING}" != "0" ]]; then
    distributed_args+=(--gradient-checkpointing)
  fi
  if [[ -n "${FSDP}" ]]; then
    distributed_args+=(--fsdp "${FSDP}")
  fi
  if [[ -n "${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP}" ]]; then
    distributed_args+=(--fsdp-transformer-layer-cls-to-wrap "${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP}")
  fi
  if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
    distributed_args+=(--deepspeed "${DEEPSPEED_CONFIG}")
  fi
  if [[ "${REWARD_SERVER_MODE}" == "paired" ]]; then
    reward_args+=(--reward-url "http://${VLLM_HOST}:${WINNER_PORT}")
  else
    reward_args+=(--winner-url "http://${VLLM_HOST}:${WINNER_PORT}" --loser-url "http://${VLLM_HOST}:${LOSER_PORT}")
  fi

  mkdir -p "${output_dir}"
  write_manifest_row "${run_name}" "${algo}" "${coef}" "${output_dir}"

  start_vllm_server "${run_name}"

  WANDB_GROUP="${WANDB_GROUP}" WANDB_NAME="${run_name}" CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    accelerate launch \
      --num_processes "${TRAIN_NUM_PROCESSES}" \
      --main_process_port "${accelerate_port}" \
      examples/token_vocab_reward_training.py \
      --trainer "${algo}" \
      --model "${POLICY_MODEL}" \
      --output-dir "${output_dir}" \
      --dataset-name "${DATASET_NAME}" \
      --dataset-split "${DATASET_SPLIT}" \
      --max-train-samples "${MAX_TRAIN_SAMPLES}" \
      "${reward_args[@]}" \
      --learning-rate "${LEARNING_RATE}" \
      --lr-scheduler-type "${LR_SCHEDULER_TYPE}" \
      --warmup-ratio "${WARMUP_RATIO}" \
      --max-grad-norm "${MAX_GRAD_NORM}" \
      --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --max-steps "${MAX_STEPS}" \
      --max-prompt-length "${MAX_PROMPT_LENGTH}" \
      --max-completion-length "${MAX_COMPLETION_LENGTH}" \
      --max-model-length "${MAX_MODEL_LEN}" \
      --num-generations "${NUM_GENERATIONS}" \
      --beta "${BETA}" \
      --seed "${SEED}" \
      --logging-steps "${LOGGING_STEPS}" \
      --save-steps "${SAVE_STEPS}" \
      --run-name "${run_name}" \
      --use-vllm \
      --vllm-mode server \
      --vllm-server-host "${VLLM_HOST}" \
      --vllm-server-port "${VLLM_PORT}" \
      --vllm-server-timeout 1800 \
      --vllm-tensor-parallel-size "${VLLM_TP}" \
      --misalignment-loss-coef "${coef}" \
      --monitor-compute-dtype float32 \
      --debug-tokenization-jsonl "${token_debug}" \
      --debug-tokenization-samples "${DEBUG_TOKENIZATION_SAMPLES}" \
      "${distributed_args[@]}" \
      --report-to wandb \
      >"${train_log}" 2>&1

  stop_vllm_server

  python scripts/summarize_experiment.py \
    --local-root "${LOCAL_ROOT}" \
    --output "${PERSIST_ROOT}/summary_metrics.csv" \
    --manifest "${PERSIST_ROOT}/manifest.jsonl"
  python scripts/analyze_misalignment_results.py \
    --summary "${PERSIST_ROOT}/summary_metrics.csv" \
    --analysis-csv "${PERSIST_ROOT}/analysis.csv" \
    --report-md "${PERSIST_ROOT}/analysis.md"
}

cat >"${PERSIST_ROOT}/run_plan.txt" <<EOF
run_id=${RUN_ID}
seed=${SEED}
wandb_project=${WANDB_PROJECT}
wandb_group=${WANDB_GROUP}
policy_model=${POLICY_MODEL}
winner_model=${WINNER_MODEL}
loser_model=${LOSER_MODEL}
dataset=${DATASET_NAME}:${DATASET_SPLIT}
local_checkpoints=${LOCAL_ROOT}
persistent_metadata=${PERSIST_ROOT}
algos=${ALGOS}
grpo_coefs=${GRPO_COEFS}
online_dpo_coefs=${ONLINE_DPO_COEFS}
max_model_len=${MAX_MODEL_LEN}
lr_scheduler_type=${LR_SCHEDULER_TYPE}
warmup_ratio=${WARMUP_RATIO}
max_grad_norm=${MAX_GRAD_NORM}
gradient_checkpointing=${GRADIENT_CHECKPOINTING}
fsdp=${FSDP}
fsdp_transformer_layer_cls_to_wrap=${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP}
deepspeed_config=${DEEPSPEED_CONFIG}
reward_server_mode=${REWARD_SERVER_MODE}
reward_server_concurrent=${REWARD_SERVER_CONCURRENT}
reward_logprob_dtype=${REWARD_LOGPROB_DTYPE}
reward_output_dtype=${REWARD_OUTPUT_DTYPE}
reward_share_gpu=${REWARD_SHARE_GPU}
debug_tokenization_samples=${DEBUG_TOKENIZATION_SAMPLES}
EOF

start_reward_servers

run_index=0
for algo in ${ALGOS}; do
  if [[ "${algo}" == "grpo" ]]; then
    coefs="${GRPO_COEFS}"
  elif [[ "${algo}" == "online_dpo" ]]; then
    coefs="${ONLINE_DPO_COEFS}"
  else
    echo "Unknown algo: ${algo}" >&2
    exit 1
  fi

  for coef in ${coefs}; do
    run_training "${algo}" "${coef}" "${run_index}"
    run_index=$((run_index + 1))
  done
done

python scripts/summarize_experiment.py \
  --local-root "${LOCAL_ROOT}" \
  --output "${PERSIST_ROOT}/summary_metrics.csv" \
  --manifest "${PERSIST_ROOT}/manifest.jsonl"
python scripts/analyze_misalignment_results.py \
  --summary "${PERSIST_ROOT}/summary_metrics.csv" \
  --analysis-csv "${PERSIST_ROOT}/analysis.csv" \
  --report-md "${PERSIST_ROOT}/analysis.md"

echo "Finished. Wandb project: ${WANDB_PROJECT}, group: ${WANDB_GROUP}"
echo "Persistent metadata: ${PERSIST_ROOT}"
echo "Local checkpoints: ${LOCAL_ROOT}"
