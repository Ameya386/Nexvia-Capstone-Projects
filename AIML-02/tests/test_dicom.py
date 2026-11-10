import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from src.dicom_utils import gray_to_dicom, save_dicom, dicom_bytes_to_gray


def test_roundtrip(tmp_path):
    img = (np.random.default_rng(0).random((128, 128)) * 255).astype(np.uint8)
    ds = gray_to_dicom(img, "DEMO0001", comments="Dataset reference label: effusion")
    save_dicom(ds, tmp_path / "a.dcm")
    gray, meta = dicom_bytes_to_gray((tmp_path / "a.dcm").read_bytes())
    assert gray.shape == (128, 128) and gray.dtype == np.uint8
    assert meta["patient_id"] == "DEMO0001" and meta["modality"] == "DX"
    assert "effusion" in meta["reference_label"]
