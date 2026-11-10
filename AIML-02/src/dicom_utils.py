"""DICOM helpers (pydicom). DICOM = image pixels + a header of metadata tags (patient, study, modality, geometry...).
Each file is one 'instance'; instances belong to a series, which belongs to a study."""
import datetime
import io
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, UID, generate_uid

DX_PRESENTATION = "1.2.840.10008.5.1.4.1.1.1.1"   # Digital X-Ray Image Storage - For Presentation


def gray_to_dicom(gray, patient_id, comments=""):
    """uint8 (H,W) array -> a valid single-frame DICOM chest X-ray (for demo studies)."""
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID(DX_PRESENTATION)
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128, is_implicit_VR=False, is_little_endian=True)
    now = datetime.datetime.now()
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientName = "DEMO^PATIENT"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.Modality = "DX"
    ds.StudyDescription = "Chest X-ray (ChestMNIST demo, not a real patient)"
    ds.ImageComments = comments
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = gray.shape
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = gray.tobytes()
    return ds


def save_dicom(ds, path):
    try:
        ds.save_as(str(path), enforce_file_format=True)      # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)     # pydicom 2.x


def dicom_bytes_to_gray(data):
    """DICOM file bytes -> (uint8 2-D image, metadata dict)."""
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:                                   # RGB (H,W,3) or multi-frame (F,H,W)
        arr = arr.mean(axis=-1) if int(ds.get("SamplesPerPixel", 1)) == 3 else arr[arr.shape[0] // 2]
    arr = arr * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = 1 - arr
    meta = {"sop_instance_uid": str(ds.SOPInstanceUID), "patient_id": str(ds.get("PatientID", "")),
            "modality": str(ds.get("Modality", "")), "description": str(ds.get("StudyDescription", "")),
            "reference_label": str(ds.get("ImageComments", ""))}
    return (arr * 255).astype(np.uint8), meta
