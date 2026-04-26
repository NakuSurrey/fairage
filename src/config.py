"""
central config for FairAge.

every magic number, every path, every hyperparameter lives here.
nothing else in the codebase should hardcode a path or a constant.
"""

from pathlib import Path

# ---------- paths ----------
# repo root resolved relative to this file — works on laptop, HPC, and Hetzner
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
UTKFACE_DIR = DATA_DIR / "utkface"
PAD_DIR = DATA_DIR / "pad"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"

# ---------- age problem setup ----------
# UTKFace ages range 0-116 — clip to 0-100 to drop noisy outliers
AGE_MIN = 0
AGE_MAX = 100
NUM_AGE_CLASSES = AGE_MAX - AGE_MIN + 1  # 101 ordinal bins

# ---------- image preprocessing ----------
IMAGE_SIZE = 224  # ResNet-50 native input size
# imagenet stats — backbone is pretrained on imagenet, must use same normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------- training hyperparameters ----------
BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# ordinal loss blend — main MAE term plus small MSE for smoother gradients
LOSS_MAE_WEIGHT = 1.0
LOSS_MSE_WEIGHT = 0.1

# ---------- bias audit groups ----------
# ethnicity codes from UTKFace filename:
# 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
ETHNICITY_LABELS = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}
GENDER_LABELS = {0: "Male", 1: "Female"}

# age buckets used in bias report
AGE_BUCKETS = [(0, 12), (13, 19), (20, 35), (36, 55), (56, 100)]

# ---------- API runtime ----------
API_HOST = "127.0.0.1"  # bind localhost only — Nginx fronts the public port
API_PORT = 8003
STREAMLIT_PORT = 8503

# inference latency target on CPU (milliseconds) — claim on README depends on this
LATENCY_TARGET_MS = 200

# ---------- random seed ----------
# fixed for reproducibility across train and eval runs
SEED = 42
