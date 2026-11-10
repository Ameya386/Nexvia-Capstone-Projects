"""Step 5: DICOM service. Pulls NEW studies from an Orthanc server (or a folder), runs inference, stores results.

  python -m service.worker --orthanc http://localhost:8042 --interval 5     # keep polling Orthanc
  python -m service.worker --orthanc http://localhost:8042 --once           # process what is there, then exit
  python -m service.worker --dicom-dir samples/dicom --once                 # no Orthanc: read .dcm files from a folder
"""
import argparse
import json
import time
from pathlib import Path
import requests
from src import config as C
from service.pipeline import ingest_dicom_bytes
from service.store import Store

META_KEY = 1024     # Orthanc user-defined metadata slot (1024-65535)


def poll_orthanc(base, store, clf, auth=None):
    since = int(store.kv_get("orthanc_since", 0))
    processed = 0
    while True:
        r = requests.get(f"{base}/changes", params={"since": since, "limit": 100}, auth=auth, timeout=30)
        r.raise_for_status()
        body = r.json()
        for ch in body["Changes"]:
            since = ch["Seq"]
            if ch["ChangeType"] != "NewInstance":
                continue
            f = requests.get(f"{base}/instances/{ch['ID']}/file", auth=auth, timeout=60)
            f.raise_for_status()
            sid, new = ingest_dicom_bytes(f.content, store, clf, source=f"orthanc:{ch['ID']}")
            if new:
                processed += 1
                study = store.get_study(sid)
                try:    # write the AI result back onto the Orthanc instance (best effort)
                    requests.put(f"{base}/instances/{ch['ID']}/metadata/{META_KEY}", auth=auth, timeout=30,
                                 data=json.dumps({"model": study["model_version"], "top_label": study["top_label"],
                                                  "confidence": study["confidence"]}))
                except requests.RequestException:
                    pass
                print(f"processed {sid}: {study['top_label']} {study['confidence']:.2f}")
        store.kv_set("orthanc_since", since)
        if body.get("Done", True):
            break
    return processed


def scan_folder(folder, store, clf):
    n = 0
    for p in sorted(Path(folder).glob("*.dcm")):
        sid, new = ingest_dicom_bytes(p.read_bytes(), store, clf, source=f"file:{p.name}")
        if new:
            n += 1
            study = store.get_study(sid)
            print(f"processed {p.name}: {study['top_label']} {study['confidence']:.2f}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orthanc", default="")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--dicom-dir", default="")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if not a.orthanc and not a.dicom_dir:
        ap.error("give --orthanc URL or --dicom-dir FOLDER")
    from src.inference import ChestClassifier
    clf, store = ChestClassifier(), Store(C.RUNTIME / "app.db")
    auth = (a.user, a.password) if a.user else None
    while True:
        try:
            n = poll_orthanc(a.orthanc.rstrip("/"), store, clf, auth) if a.orthanc else scan_folder(a.dicom_dir, store, clf)
            print(f"cycle done, {n} new studies")
        except requests.RequestException as e:
            print("Orthanc not reachable yet:", e)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
