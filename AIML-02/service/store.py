"""SQLite store for studies, AI results and a tamper-evident, append-only audit log.
Every audit row stores SHA-256(previous hash + row contents) so any later edit breaks the chain (verify_audit_chain).
SQLite triggers also refuse UPDATE/DELETE on the audit table."""
import datetime
import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
  id TEXT PRIMARY KEY, source TEXT, patient_id TEXT, modality TEXT, description TEXT, reference_label TEXT,
  created_at TEXT, model_version TEXT, top_label TEXT, confidence REAL, probs_json TEXT,
  status TEXT NOT NULL DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, study_id TEXT, action TEXT NOT NULL, actor TEXT NOT NULL,
  comment TEXT, model_version TEXT, top_label TEXT, confidence REAL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""
DECISIONS = {"approve": "approved", "reject": "rejected"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _digest(prev, ts, study_id, action, actor, comment, model_version, top_label, confidence):
    payload = json.dumps([prev, ts, study_id, action, actor, comment, model_version, top_label, confidence],
                         separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ---------- audit
    def _append_audit(self, c, study_id, action, actor, comment="", model_version="", top_label="", confidence=None):
        row = c.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else "GENESIS"
        ts = now()
        h = _digest(prev, ts, study_id, action, actor, comment, model_version, top_label, confidence)
        c.execute("INSERT INTO audit_log (ts, study_id, action, actor, comment, model_version, top_label, confidence,"
                  " prev_hash, hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (ts, study_id, action, actor, comment, model_version, top_label, confidence, prev, h))

    def audit(self, limit=200):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))]

    def verify_audit_chain(self):
        with self._conn() as c:
            prev = "GENESIS"
            for r in c.execute("SELECT * FROM audit_log ORDER BY id"):
                expect = _digest(prev, r["ts"], r["study_id"], r["action"], r["actor"], r["comment"],
                                 r["model_version"], r["top_label"], r["confidence"])
                if r["prev_hash"] != prev or r["hash"] != expect:
                    return False
                prev = r["hash"]
        return True

    # ---------- studies
    def add_study(self, rec):
        """rec keys: id, source, patient_id, modality, description, reference_label, model_version, top_label,
        confidence, probs (dict). Returns False if the study already exists."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            cur = c.execute(
                "INSERT OR IGNORE INTO studies (id, source, patient_id, modality, description, reference_label,"
                " created_at, model_version, top_label, confidence, probs_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (rec["id"], rec["source"], rec["patient_id"], rec["modality"], rec["description"],
                 rec["reference_label"], now(), rec["model_version"], rec["top_label"], rec["confidence"],
                 json.dumps(rec["probs"])))
            if cur.rowcount == 0:
                return False
            self._append_audit(c, rec["id"], "inference_completed", "system", "", rec["model_version"],
                               rec["top_label"], rec["confidence"])
            return True

    def list_studies(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT id, patient_id, modality, description, top_label, confidence, status, created_at"
                " FROM studies ORDER BY created_at DESC, id")]

    def get_study(self, study_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["probs"] = json.loads(d.pop("probs_json"))
        return d

    def record_decision(self, study_id, decision, reviewer, comment=""):
        if decision not in DECISIONS:
            raise ValueError("decision must be 'approve' or 'reject'")
        reviewer = (reviewer or "").strip()
        if not reviewer:
            raise ValueError("reviewer name is required")
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            s = c.execute("SELECT * FROM studies WHERE id=?", (study_id,)).fetchone()
            if not s:
                raise KeyError(study_id)
            c.execute("UPDATE studies SET status=? WHERE id=?", (DECISIONS[decision], study_id))
            self._append_audit(c, study_id, decision, reviewer, comment or "", s["model_version"],
                               s["top_label"], s["confidence"])

    # ---------- key/value (worker bookmark)
    def kv_get(self, key, default=None):
        with self._conn() as c:
            r = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def kv_set(self, key, value):
        with self._conn() as c:
            c.execute("INSERT INTO kv (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, str(value)))
