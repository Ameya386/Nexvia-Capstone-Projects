"""Backend for the radiologist viewer.   uvicorn service.api:app --port 8000"""
import json
import re
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src import config as C
from service.store import Store

app = FastAPI(title="AI-Assisted Imaging Review - " + C.DISCLAIMER)
store = Store(C.RUNTIME / "app.db")
SAFE_ID = re.compile(r"^[0-9A-Za-z._-]+$")


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str
    comment: str = ""


def _study_or_404(study_id):
    if not SAFE_ID.match(study_id):
        raise HTTPException(400, "bad id")
    s = store.get_study(study_id)
    if not s:
        raise HTTPException(404, "study not found")
    return s


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


@app.get("/api/health")
def health():
    return {"status": "ok", "notice": C.DISCLAIMER}


@app.get("/api/studies")
def studies():
    return store.list_studies()


@app.get("/api/studies/{study_id}")
def study(study_id: str):
    return _study_or_404(study_id)


@app.get("/api/studies/{study_id}/image")
def image(study_id: str):
    _study_or_404(study_id)
    return FileResponse(C.RUNTIME / "images" / f"{study_id}_orig.png")


@app.get("/api/studies/{study_id}/overlay")
def overlay(study_id: str):
    _study_or_404(study_id)
    return FileResponse(C.RUNTIME / "images" / f"{study_id}_cam.png")


@app.post("/api/studies/{study_id}/decision")
def decide(study_id: str, d: Decision):
    _study_or_404(study_id)
    try:
        store.record_decision(study_id, d.decision, d.reviewer, d.comment)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return store.get_study(study_id)


@app.get("/api/audit")
def audit():
    return {"chain_valid": store.verify_audit_chain(), "entries": store.audit()}


@app.get("/api/segmentation")
def segmentation():
    d = C.RESULTS / "seg_examples"
    return {"metrics": _read_json(C.RESULTS / "seg_metrics.json"),
            "images": sorted(p.name for p in d.glob("*.png")) if d.exists() else []}


@app.get("/api/segmentation/{name}")
def seg_image(name: str):
    if not SAFE_ID.match(name) or not (C.RESULTS / "seg_examples" / name).exists():
        raise HTTPException(404, "not found")
    return FileResponse(C.RESULTS / "seg_examples" / name)


@app.get("/api/model")
def model_info():
    return {"notice": C.DISCLAIMER, "meta": _read_json(C.MODELS / "classifier_meta.json"),
            "metrics": _read_json(C.RESULTS / "classifier_metrics.json")}


app.mount("/", StaticFiles(directory=str(C.FRONTEND), html=True), name="frontend")
