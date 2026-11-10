"""Central paths, labels and params."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = DATA / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RUNTIME = ROOT / "runtime"
SAMPLES = ROOT / "samples"
FRONTEND = ROOT / "frontend"

# ChestMNIST = NIH ChestX-ray14 (14 findings, multi-label)
LABELS = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule", "pneumonia",
          "pneumothorax", "consolidation", "edema", "emphysema", "fibrosis", "pleural thickening", "hernia"]

MODEL_VERSION = "densenet121-chestmnist128-v1"
CAM_LAYER = "features.denseblock3"      # 8x8 feature map at 128px input -> smoother heatmaps than the last block
DISCLAIMER = "FOR EDUCATION ONLY - NOT FOR CLINICAL USE"


def load_params():
    import yaml
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
