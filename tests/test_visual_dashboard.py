"""Charts must preserve missing data, measurement scope and workspace isolation."""

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from backend.identity import as_user
from backend.storage import Store
from dashboard.charts import METRICS, metric_chart, question_heatmap, tradeoff_chart


def measured_run():
    configs = [{"id": "a", "name": "Baseline"}, {"id": "b", "name": "Reranking"}]
    return {
        "id": "fixture",
        "status": "completed",
        "created_at": "2026-10-07T10:00:00Z",
        "completed": 4,
        "total": 4,
        "configurations": configs,
        "questions": [{"question": "First?"}, {"question": "Second?"}],
        "summary": [
            {
                "name": c["name"],
                "configuration_id": c["id"],
                **dict.fromkeys(METRICS, 0.8),
                "overall": 0.8,
                "complete": True,
                "expected_questions": 2,
            }
            for c in configs
        ],
        "rows": [
            {
                "configuration_id": c["id"],
                "question_index": q,
                "answer": "Saved answer",
                "scores": dict.fromkeys(METRICS, 0.8),
                "latency_seconds": q + 1,
            }
            for c in configs
            for q in range(2)
        ],
    }


def test_heatmap_keeps_missing_cells_and_negative_scores():
    run = measured_run()
    run["rows"][0]["scores"]["answer_relevancy"] = None
    run["rows"][1]["scores"]["answer_relevancy"] = -0.3
    cells = question_heatmap(run, "answer_relevancy").data[0]
    assert list(cells.z[0]) == [None, -0.3]
    assert cells.zmin == -0.3 and cells.hoverongaps is False
    run["summary"][0]["faithfulness"] = None
    figure = metric_chart(run["summary"], "faithfulness")
    assert len(figure.data) == 1 and figure.data[0].name == "Reranking"
    assert figure.data[0].marker.color == "#169b8a"


def test_tradeoff_requires_complete_measurements():
    run = measured_run()
    figure = tradeoff_chart(run)
    assert len(figure.data) == 2 and figure.data[0].x == (1.5,)
    run["rows"][0]["latency_seconds"] = None
    assert len(tradeoff_chart(run).data) == 1
    run["summary"][1]["complete"] = False
    assert not tradeoff_chart(run).data


def test_long_configuration_names_do_not_merge_chart_categories():
    run = measured_run()
    for i, config in enumerate(run["configurations"]):
        config["name"] = "A shared configuration name prefix " + str(i)
        run["summary"][i]["name"] = config["name"]
    bars = metric_chart(run["summary"], "faithfulness")
    assert [trace.y[0] for trace in bars.data] == ["a", "b"]
    assert question_heatmap(run, "faithfulness").data[0].y == ("a", "b")
    run = measured_run()
    run["status"] = "partial"
    assert not tradeoff_chart(run).data


def test_summary_projection_is_compact_and_private(tmp_path):
    store = Store(tmp_path)
    run = measured_run()
    run["profiling"] = {
        "summary": {"groq": {"http_requests": 7}},
        "attempts": [{"secret": "private"}],
    }
    with as_user("alice"):
        store.save("run", run)
        projection = store.run_history(include_summary=True)[0]
        assert projection["summary"] == run["summary"]
        assert projection["profile_summary"] == run["profiling"]["summary"]
        assert (
            "rows" not in projection
            and "questions" not in projection
            and "profiling" not in projection
        )
        assert "summary" not in store.run_history()[0]
    with as_user("bob"):
        assert store.run_history(include_summary=True) == []


@pytest.mark.parametrize("runs", [[], [measured_run()]])
def test_overview_uses_only_one_read_and_opens_setup(monkeypatch, runs):
    monkeypatch.setenv("BACKEND_URL", "http://fixture")

    class Response:
        ok = True

        def json(self):
            return {"runs": runs, "next_offset": None}

    with patch("requests.request", return_value=Response()) as request:
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "dashboard/app.py"), default_timeout=20
        ).run()
        assert not app.exception
        assert app.radio(key="page").value == "Overview"
        request.assert_called_once()
        assert request.call_args.args[0] == "GET"
        assert request.call_args.kwargs["params"] == {"limit": 50, "include_summary": True}
        if runs:
            assert len(app.get("plotly_chart")) == 3
            assert {m.label: m.value for m in app.metric}["Recorded HTTP attempts"] == "—"
        next(b for b in app.button if b.label == "New benchmark").click().run()
        assert not app.exception and app.radio(key="page").value == "Upload / Setup"
