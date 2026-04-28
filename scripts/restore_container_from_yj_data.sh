#!/usr/bin/env bash
set -euo pipefail

# Restore the local experiment workspace after a container restart.
#
# Usage:
#   scripts/restore_container_from_yj_data.sh [backup_root]
#
# If backup_root is omitted, the newest
# /yj_data/trl_misalignment/container_backup_* directory is used.

BACKUP_ROOT="${1:-}"
if [[ -z "${BACKUP_ROOT}" ]]; then
  BACKUP_ROOT="$(find /yj_data/trl_misalignment -maxdepth 1 -type d -name 'container_backup_*' | sort | tail -n 1)"
fi

if [[ -z "${BACKUP_ROOT}" || ! -d "${BACKUP_ROOT}" ]]; then
  echo "No backup root found. Pass /yj_data/trl_misalignment/container_backup_... explicitly." >&2
  exit 1
fi

REPO_DIR="${REPO_DIR:-/root/trl-misalignment}"
LOGPROB_DIR="${LOGPROB_DIR:-/root/logprob-engine}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[restore] backup root: ${BACKUP_ROOT}"

mkdir -p /root/.cache

if [[ -d "${BACKUP_ROOT}/root/trl-misalignment" ]]; then
  mkdir -p "${REPO_DIR}"
  rsync -a --info=progress2 "${BACKUP_ROOT}/root/trl-misalignment/" "${REPO_DIR}/"
else
  git clone git@github.com:yunjae-won/trl-misalignment.git "${REPO_DIR}"
fi

if [[ -d "${BACKUP_ROOT}/root/logprob-engine" ]]; then
  mkdir -p "${LOGPROB_DIR}"
  rsync -a --info=progress2 "${BACKUP_ROOT}/root/logprob-engine/" "${LOGPROB_DIR}/"
fi

if [[ -d "${BACKUP_ROOT}/root/.cache/huggingface" ]]; then
  mkdir -p /root/.cache/huggingface
  rsync -a --info=progress2 "${BACKUP_ROOT}/root/.cache/huggingface/" /root/.cache/huggingface/
fi

if [[ -d "${BACKUP_ROOT}/root/.cache/pip" ]]; then
  mkdir -p /root/.cache/pip
  rsync -a --info=progress2 "${BACKUP_ROOT}/root/.cache/pip/" /root/.cache/pip/
fi

if [[ -d "${BACKUP_ROOT}/private_root_home" ]]; then
  rsync -a "${BACKUP_ROOT}/private_root_home/" /root/
  chmod 700 /root/.ssh 2>/dev/null || true
  chmod 600 /root/.ssh/* /root/.netrc 2>/dev/null || true
fi

cd "${REPO_DIR}"

if [[ "${RESTORE_EXACT_PIP:-0}" == "1" && -f "${BACKUP_ROOT}/manifests/pip-freeze-all.txt" ]]; then
  "${PYTHON_BIN}" -m pip install -r "${BACKUP_ROOT}/manifests/pip-freeze-all.txt"
else
  "${PYTHON_BIN}" -m pip install -e .
  if [[ -d "${LOGPROB_DIR}" ]]; then
    "${PYTHON_BIN}" -m pip install -e "${LOGPROB_DIR}"
  fi
fi

echo "[restore] repo commit: $(git -C "${REPO_DIR}" rev-parse HEAD)"
echo "[restore] done"
