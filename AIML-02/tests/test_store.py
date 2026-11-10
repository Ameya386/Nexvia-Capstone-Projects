import sqlite3
import pytest
from service.store import Store


def rec(i):
    return {"id": f"1.2.3.{i}", "source": "t", "patient_id": f"P{i}", "modality": "DX", "description": "d",
            "reference_label": "x", "model_version": "v1", "top_label": "effusion", "confidence": 0.4, "probs": {"a": 0.1}}


def test_add_and_decide(tmp_path):
    s = Store(tmp_path / "t.db")
    assert s.add_study(rec(1)) is True
    assert s.add_study(rec(1)) is False            # duplicates ignored
    s.record_decision("1.2.3.1", "approve", "Dr A", "looks fine")
    assert s.get_study("1.2.3.1")["status"] == "approved"
    assert [a["action"] for a in s.audit()] == ["approve", "inference_completed"]
    assert s.verify_audit_chain()


def test_validation(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add_study(rec(1))
    with pytest.raises(ValueError):
        s.record_decision("1.2.3.1", "maybe", "Dr A")
    with pytest.raises(ValueError):
        s.record_decision("1.2.3.1", "reject", "  ")
    with pytest.raises(KeyError):
        s.record_decision("nope", "reject", "Dr A")


def test_audit_is_append_only_and_tamper_evident(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add_study(rec(1))
    s.record_decision("1.2.3.1", "reject", "Dr A", "artefact")
    with sqlite3.connect(s.path) as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("UPDATE audit_log SET comment='edited' WHERE id=1")
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("DELETE FROM audit_log WHERE id=1")
    # simulate someone bypassing the triggers: the hash chain must still catch it
    with sqlite3.connect(s.path) as c:
        c.execute("DROP TRIGGER audit_no_update")
        c.execute("UPDATE audit_log SET comment='edited' WHERE id=2")
    assert s.verify_audit_chain() is False
