import asyncio
import io
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pypdf import PdfWriter
from backend.models import Configuration, RunRequest, METRICS
from backend.ingestion import extract_document, parse_test_set
from backend.storage import Store


def test_reject_bad_configuration():
    with pytest.raises(ValidationError):
        Configuration(name="invalid", chunk_size=64, overlap=64)
    with pytest.raises(ValidationError):
        Configuration(name="invalid", embedding_model="arbitrary/model")
    with pytest.raises(ValidationError):
        RunRequest(document_set_id="x", test_set_id="y", configuration_ids=["same", "same"])


def test_groq_judge_uses_separate_completions(monkeypatch):
    from backend.pipeline import make_llm, make_metrics
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    llm = make_llm()
    metrics = make_metrics(llm, None)
    assert metrics["answer_relevancy"].llm.bypass_n is True
    assert metrics["answer_relevancy"].strictness == 3
    assert llm.n == 1


def test_groq_judge_combines_text_without_batch_usage_bug(monkeypatch):
    from backend.groq_judge import GroqRagasLLM
    from ragas.llms.base import LangchainLLMWrapper
    from langchain_core.outputs import LLMResult, Generation
    from backend.pipeline import make_llm
    monkeypatch.setenv("GROQ_API_KEY", "test-placeholder")
    calls = []

    async def one_completion(self, prompt, n, **kwargs):
        calls.append(n)
        return LLMResult(generations=[[Generation(text="test")]],
                         llm_output={"token_usage": {"completion_tokens_details": {"reasoning_tokens": 10}}})

    monkeypatch.setattr(LangchainLLMWrapper, "agenerate_text", one_completion)
    result = asyncio.run(GroqRagasLLM(make_llm(), bypass_n=True).agenerate_text("test", n=3))
    assert calls == [1, 1, 1]
    assert len(result.generations[0]) == 3


def test_ingestion_and_test_formats():
    docs = extract_document("../../test.txt", b"A useful handbook.")
    assert docs[0]["source"] == "test.txt"
    assert parse_test_set("test.csv", b'question,expected_answer\nWhat is this?,A handbook\n').questions[0].reference == "A handbook"
    assert len(parse_test_set("test.json", b'[{"question":"What?","reference":"Answer"}]').questions) == 1
    with pytest.raises(ValueError):
        extract_document("empty.txt", b"")
    with pytest.raises(ValueError):
        parse_test_set("bad.json", b'["invalid"]')


def test_blank_and_encrypted_pdf_rejected():
    pdf = PdfWriter()
    pdf.add_blank_page(width=100, height=100)
    raw = io.BytesIO()
    pdf.write(raw)
    with pytest.raises(ValueError, match="No extractable"):
        extract_document("scan.pdf", raw.getvalue())
    pdf.encrypt("password")
    raw = io.BytesIO()
    pdf.write(raw)
    with pytest.raises(ValueError, match="Encrypted"):
        extract_document("private.pdf", raw.getvalue())


def test_api_upload_demo_auth_and_missing_key(tmp_path, monkeypatch):
    import backend.main as main
    monkeypatch.setattr(main, "store", Store(tmp_path))
    monkeypatch.setenv("API_TOKEN", "test-only")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with TestClient(main.app) as client:
        assert client.post("/api/demo").status_code == 401
        client.headers["Authorization"] = "Bearer test-only"
        demo = client.post("/api/demo").json()
        assert len(demo["questions"]) == 10
        assert len(demo["configurations"]) == 2
        assert client.post("/api/documents", files={"files": ("hello.txt", b"hello world")}).status_code == 201
        assert client.post("/api/documents", files={"files": ("bad.exe", b"bad")}).status_code == 422
        body = {k: demo[k] for k in ["document_set_id", "test_set_id", "configuration_ids"]}
        assert client.post("/api/runs", json=body).status_code == 503
        assert client.get("/api/runs/unknown").status_code == 404


def test_metric_failures_not_zeroed_and_summary_excludes_incomplete():
    from backend.pipeline import score_row, summarize

    class BrokenMetric:
        async def single_turn_ascore(self, sample, **kwargs):
            return float("nan")

    row = {"question": "What is this?", "answer": "A lab", "reference": "A lab", "contexts": [{"text": "A lab"}]}
    scores, errors = asyncio.run(score_row({m: BrokenMetric() for m in METRICS}, row))
    assert all(value is None for value in scores.values())
    assert set(errors) == set(METRICS)
    rows = [{"configuration_id": "a", "scores": {m: 0.8 for m in METRICS}},
            {"configuration_id": "a", "scores": scores}]
    summary = summarize(rows, [{"id": "a", "name": "A"}], 2)[0]
    assert summary["faithfulness"] == 0.8
    assert summary["overall"] is None
    assert summary["valid_counts"]["faithfulness"] == 1


def test_restart_preserves_partial_data_and_csv_escapes(tmp_path, monkeypatch):
    import backend.main as main
    store = Store(tmp_path)
    monkeypatch.setattr(main, "store", store)
    monkeypatch.delenv("API_TOKEN", raising=False)
    row = {"question": "=HYPERLINK(\"evil\")", "scores": {m: 0.5 for m in METRICS}, "contexts": [], "errors": {}}
    run = store.save("run", {"status": "running", "rows": [row]})
    with TestClient(main.app) as client:
        assert client.get(f"/api/runs/{run['id']}").json()["status"] == "failed"
        csv = client.get(f"/api/runs/{run['id']}/export")
        assert csv.status_code == 200
        assert "'=HYPERLINK" in csv.text
