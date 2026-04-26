#!/bin/bash
#SBATCH --job-name=fairage-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# submitted with: sbatch slurm/train_age.sh
# run from the repo root on the login node — sbatch's CWD is propagated to the job.

set -euo pipefail

echo "[job] starting at $(date)"
echo "[job] SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "[job] hostname = $(hostname)"
echo "[job] cwd = $(pwd)"

# ---------- env ----------
module load Anaconda3/2024.02-1
module load CUDA/12.2.2

# conda activate is a shell function — non-interactive shells need this hook first
eval "$(conda shell.bash hook)"
conda activate fairage

# python buffers stdout in non-interactive shells — flush every line so we see live progress
export PYTHONUNBUFFERED=1

# show the gpu we got — useful when MIG slice changes between jobs
nvidia-smi -L || true

# ---------- run training ----------
python -m src.training.train_age "$@"

echo "[job] done at $(date)"
