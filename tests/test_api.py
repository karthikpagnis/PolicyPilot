"""API tests for claim-pipeline."""

from types import SimpleNamespace

import fitz
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def _valid_pdf_bytes() -> bytes:
    doc = fitz.open()
    doc.new_page(width=300, height=300)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_root_health() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"


def test_process_rejects_non_pdf() -> None:
    response = client.post(
        "/api/process",
        data={"claim_id": "CLM-001"},
        files={"file": ("not_a_pdf.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF files are accepted" in response.json()["detail"]


def test_process_rejects_empty_pdf() -> None:
    response = client.post(
        "/api/process",
        data={"claim_id": "CLM-002"},
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded PDF file is empty."


def test_process_success_with_mocked_workflow(monkeypatch) -> None:
    async def fake_ainvoke(state: dict) -> dict:
        assert state["claim_id"] == "CLM-003"
        return {"final_result": {"claim_id": state["claim_id"], "ok": True}}

    monkeypatch.setattr(main, "claim_workflow", SimpleNamespace(ainvoke=fake_ainvoke))

    response = client.post(
        "/api/process",
        data={"claim_id": "CLM-003"},
        files={"file": ("sample.pdf", _valid_pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json() == {"claim_id": "CLM-003", "ok": True}


def test_process_returns_500_when_workflow_raises(monkeypatch) -> None:
    async def failing_ainvoke(state: dict) -> dict:
        raise RuntimeError("workflow exploded")

    monkeypatch.setattr(main, "claim_workflow", SimpleNamespace(ainvoke=failing_ainvoke))

    response = client.post(
        "/api/process",
        data={"claim_id": "CLM-004"},
        files={"file": ("sample.pdf", _valid_pdf_bytes(), "application/pdf")},
    )
    assert response.status_code == 500
    assert "Pipeline failed:" in response.json()["detail"]
