"""Loads the trained classifier and returns prediction + Grad-CAM overlay for one image."""
import numpy as np
import torch
from src import config as C
from src.model import build_model
from src.preprocess import normalize_batch
from src.gradcam import GradCAM, overlay


class ChestClassifier:
    def __init__(self, weights=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(pretrained=False)
        state = torch.load(weights or (C.MODELS / "densenet121_chestmnist.pt"), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.cam = GradCAM(self.model, C.CAM_LAYER)
        self.size = C.load_params()["data"]["image_size"]

    def predict(self, gray):
        """gray: uint8 (size,size). Returns dict with probabilities, top finding, confidence, heatmap overlay."""
        x = normalize_batch(torch.from_numpy(np.array(gray)).float().div(255)[None, None].to(self.device))
        cam, probs, top = self.cam(x)
        return {"probs": {n: float(p) for n, p in zip(C.LABELS, probs)},
                "top_label": C.LABELS[top], "confidence": float(probs[top]),
                "cam": cam, "overlay": overlay(gray, cam)}
