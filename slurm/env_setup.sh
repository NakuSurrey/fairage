#!/bin/bash
# one-time setup for FairAge on Surrey HPC.
# run once on the login node before submitting any training job:
#   bash slurm/env_setup.sh
#
# safe to re-run — every step checks if work is already done.
# wrap with `screen -S setup` so an SSH drop does not kill the install.

set -euo pipefail

echo "[env_setup] starting at $(date)"
echo "[env_setup] host = $(hostname)"

# ---------- modules ----------
echo "[env_setup] loading cluster modules"
module load Anaconda3/2024.02-1
module load CUDA/12.2.2

# ---------- conda env ----------
ENV_NAME="fairage"

# load conda activate as a function — non-interactive shells skip this by default
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^${ENV_NAME}\\s"; then
    echo "[env_setup] conda env '${ENV_NAME}' already exists, skipping create"
else
    echo "[env_setup] creating conda env '${ENV_NAME}'"
    conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
echo "[env_setup] using python: $(which python)"

# ---------- python deps ----------
echo "[env_setup] installing python deps"
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-train.txt

# ---------- UTKFace download ----------
DATA_DIR="data/utkface"
if [ -d "${DATA_DIR}" ] && [ "$(ls -A ${DATA_DIR} 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "[env_setup] UTKFace already present in ${DATA_DIR}, skipping download"
else
    echo "[env_setup] UTKFace not found — see data/README.md for download steps"
    echo "[env_setup] expected: kaggle download or manual upload, then unzip into ${DATA_DIR}"
fi

# ---------- W&B login (optional) ----------
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[env_setup] logging in to Weights & Biases"
    wandb login --relogin "${WANDB_API_KEY}"
else
    echo "[env_setup] WANDB_API_KEY not set — training will skip W&B logging"
    echo "[env_setup] export WANDB_API_KEY in ~/.bashrc to enable"
fi

echo "[env_setup] done at $(date)"
echo "[env_setup] activate the env in future sessions with:"
echo "  module load Anaconda3/2024.02-1 CUDA/12.2.2"
echo "  conda activate ${ENV_NAME}"
