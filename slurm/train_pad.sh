#!/bin/bash
# SLURM job — train the PAD detector on Surrey HPC.
# submit from the repo root on the HPC login node:
#     sbatch slurm/train_pad.sh
#
# expects:
#   - slurm/env_setup.sh has been run once to create the conda env
#   - data/pad/ contains NUAA real/ and attack/ subfolders (or NUAA native names)
#   - WANDB_API_KEY in the env if W&B logging is wanted (optional)

#SBATCH --job-name=fairage-pad
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# python buffering off so logs arrive in real time, not at job end
export PYTHONUNBUFFERED=1

# load HPC modules — same versions used for the age model job
module purge
module load cuda/12.1
module load anaconda3

# conda activation in non-interactive shell needs the explicit hook
eval "$(conda shell.bash hook)"
conda activate fairage

# move to the repo root — the script must be run from there
cd "$SLURM_SUBMIT_DIR"

# log basic context — useful when reading slurm-<jobid>.out later
echo "===== PAD training job ====="
echo "host:        $(hostname)"
echo "date:        $(date)"
echo "cwd:         $(pwd)"
echo "git head:    $(git rev-parse --short HEAD)"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "python:      $(which python)"
echo "============================"

# run the training script — flags can be overridden by editing this file
python -m src.training.train_pad \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 4

echo "===== job done ====="
