"""Run the whole pipeline without DVC:  python run_all.py   (DVC equivalent: dvc repro)"""
import subprocess
import sys

for m in ("src.prepare_data", "src.train_classifier", "src.train_segmentation", "src.make_examples"):
    print("\n>>>", m)
    subprocess.run([sys.executable, "-m", m], check=True)
print("\nDone. Next: python -m service.worker --dicom-dir samples/dicom --once ; uvicorn service.api:app --port 8000")
