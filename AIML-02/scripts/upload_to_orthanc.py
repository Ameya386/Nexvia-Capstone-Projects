"""Uploads every .dcm in a folder to an Orthanc server (simulates a scanner / PACS sending studies).
   python scripts/upload_to_orthanc.py --dir samples/dicom --url http://localhost:8042"""
import argparse
from pathlib import Path
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="samples/dicom")
ap.add_argument("--url", default="http://localhost:8042")
a = ap.parse_args()
files = sorted(Path(a.dir).glob("*.dcm"))
for p in files:
    r = requests.post(a.url.rstrip("/") + "/instances", data=p.read_bytes(),
                      headers={"Content-Type": "application/dicom"}, timeout=60)
    r.raise_for_status()
    print(p.name, "->", r.json().get("Status", "ok"))
print(f"uploaded {len(files)} files")
