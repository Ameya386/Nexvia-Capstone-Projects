"""Central config. Import this FIRST in every script (it sets TF_USE_LEGACY_KERAS)."""
import os
from pathlib import Path

# TensorFlow Recommenders is built on Keras 2 -> use tf-keras with TF >= 2.16
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"
RESULTS = ROOT / "results"
MODEL_REPO = ROOT / "model_repository"

BASE = "https://files.grouplens.org/datasets/movielens/"
DATASETS = {
    "small": ("ml-latest-small", BASE + "ml-latest-small.zip"),  # 100k ratings - fast, for dev/demo
    "25m": ("ml-25m", BASE + "ml-25m.zip"),                      # 25M ratings - the "large-scale" run
}

POS_THRESHOLD = 4.0     # rating >= 4 counts as a positive ("liked / clicked")
TRAIN_Q, VAL_Q = 0.80, 0.90   # time-based split by timestamp quantile (NOT random)
MIN_HISTORY = 3         # users with fewer train interactions are treated as cold-start

# retrieval
EMB_DIM = 64
RETRIEVAL_EPOCHS = 5
RETRIEVAL_BATCH = 4096
TEMPERATURE = 0.05
N_CANDIDATES = 200      # stage-1 returns this many candidates

# ranking
NEG_RATIO = 2           # sampled negatives per positive for the ranker
RANK_EPOCHS = 6
RANK_BATCH = 2048

# serving
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = 3600
TRITON_URL = os.getenv("TRITON_URL", "")   # e.g. localhost:8000 ; empty = score locally

SEED = 42


def ensure_dirs():
    for p in (RAW, PROC, ART, RESULTS, MODEL_REPO):
        p.mkdir(parents=True, exist_ok=True)
