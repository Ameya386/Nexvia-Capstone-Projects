"""Turns one DICOM file into a stored AI result (probabilities + Grad-CAM overlay)."""
import numpy as np
from PIL import Image
from src import config as C
from src.dicom_utils import dicom_bytes_to_gray
from src.preprocess import resize_gray

IMG_DIR = C.RUNTIME / "images"
VIEW_SIZE = 384
RESAMPLE = getattr(Image, "Resampling", Image).BICUBIC


def _save_png(arr, path):
    Image.fromarray(arr).resize((VIEW_SIZE, VIEW_SIZE), RESAMPLE).save(path)


def ingest_dicom_bytes(data, store, clf, source="folder"):
    """Returns (study_id, was_new)."""
    gray, meta = dicom_bytes_to_gray(data)
    study_id = meta["sop_instance_uid"]
    if store.get_study(study_id):
        return study_id, False
    r = clf.predict(resize_gray(gray, clf.size))
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _save_png(resize_gray(gray, clf.size), IMG_DIR / f"{study_id}_orig.png")
    _save_png(r["overlay"], IMG_DIR / f"{study_id}_cam.png")
    new = store.add_study({"id": study_id, "source": source, "patient_id": meta["patient_id"],
                           "modality": meta["modality"], "description": meta["description"],
                           "reference_label": meta["reference_label"], "model_version": C.MODEL_VERSION,
                           "top_label": r["top_label"], "confidence": r["confidence"], "probs": r["probs"]})
    return study_id, new
