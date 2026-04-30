import asyncio

import pandas as pd
from starlette.requests import Request

from app.routes.analyze import (
    ANALYSIS_STORE,
    DATASET_STORE,
    FollowUpRequest,
    answer_follow_up_question,
    follow_up,
)
from app.services import charting
from app.services.profiler import profile_dataframe


def _fake_save_figure(fig, suffix: str) -> str:
    charting.plt.close(fig)
    return f"/test/{suffix}.png"


def _iris_like_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6],
            "SepalLengthCm": [5.1, 4.9, 4.7, 6.0, 6.4, 6.9],
            "SepalWidthCm": [3.5, 3.0, 3.2, 2.2, 3.2, 3.1],
            "Species": [
                "Iris-setosa",
                "Iris-setosa",
                "Iris-setosa",
                "Iris-versicolor",
                "Iris-versicolor",
                "Iris-virginica",
            ],
        }
    )


def test_profile_marks_iris_id_as_identifier() -> None:
    profile = profile_dataframe(_iris_like_df())

    assert profile["semantic_columns"]["Id"] == "identifier"
    assert profile["semantic_columns"]["SepalLengthCm"] == "numeric_feature"
    assert profile["semantic_columns"]["Species"] == "categorical"
    assert "Id" in profile["identifier_columns"]


def test_profile_marks_unique_id_columns_as_identifier() -> None:
    df = pd.DataFrame(
        {
            "order_id": ["A001", "A002", "A003"],
            "customer_id": [101, 102, 103],
            "amount": [10.0, 15.0, 20.0],
        }
    )

    profile = profile_dataframe(df)

    assert profile["semantic_columns"]["order_id"] == "identifier"
    assert profile["semantic_columns"]["customer_id"] == "identifier"
    assert profile["semantic_columns"]["amount"] == "measure"
    assert profile["semantic_profile"]["order_id"]["confidence"] >= 0.7
    assert profile["semantic_profile"]["order_id"]["reasons"]


def test_profile_marks_common_code_columns_as_identifiers() -> None:
    df = pd.DataFrame(
        {
            "invoice_no": ["INV-001", "INV-002", "INV-003"],
            "sku": ["SKU-001", "SKU-002", "SKU-003"],
            "transaction_number": [1001, 1002, 1003],
            "revenue": [10.0, 15.0, 20.0],
        }
    )

    profile = profile_dataframe(df)

    assert profile["semantic_columns"]["invoice_no"] == "identifier"
    assert profile["semantic_columns"]["sku"] == "identifier"
    assert profile["semantic_columns"]["transaction_number"] == "identifier"
    assert profile["semantic_columns"]["revenue"] == "measure"


def test_profile_does_not_treat_valid_as_id_name() -> None:
    df = pd.DataFrame({"valid": [1, 0, 1], "amount": [10.0, 15.0, 20.0]})

    profile = profile_dataframe(df)

    assert profile["semantic_columns"]["valid"] != "identifier"


def test_default_charts_skip_id_histogram_and_count_species(monkeypatch) -> None:
    monkeypatch.setattr(charting, "_save_figure", _fake_save_figure)

    charts = charting.generate_charts(_iris_like_df(), column_map={})

    histogram = next(chart for chart in charts if chart["type"] == "histogram")
    bar = next(chart for chart in charts if chart["type"] == "bar")

    assert histogram["column"] == "SepalLengthCm"
    assert histogram["column"] != "Id"
    assert any(
        "Identifier columns were skipped" in warning
        for warning in histogram["warnings"]
    )
    assert bar["column"] == "Species"
    assert bar["title"] == "Record Count by Species"


def test_histogram_request_for_id_converts_to_sequence_scatter(monkeypatch) -> None:
    monkeypatch.setattr(charting, "_save_figure", _fake_save_figure)

    charts = charting.generate_charts(
        _iris_like_df(),
        column_map={},
        chart_requests=[{"type": "histogram", "column": "Id"}],
    )

    chart = charts[0]

    assert chart["type"] == "scatter"
    assert chart["x_column"] == "__row_index"
    assert chart["y_column"] == "Id"
    assert any(
        "histogram was not selected" in warning
        for warning in chart["warnings"]
    )


def test_explicit_sequence_scatter_uses_row_order_for_id(monkeypatch) -> None:
    monkeypatch.setattr(charting, "_save_figure", _fake_save_figure)

    charts = charting.generate_charts(
        _iris_like_df(),
        column_map={},
        chart_requests=[
            {"type": "scatter", "x": "__row_index", "y": "Id", "intent": "sequence"}
        ],
    )

    chart = charts[0]

    assert chart["type"] == "scatter"
    assert chart["x_column"] == "__row_index"
    assert chart["y_column"] == "Id"
    assert chart["rows"][0] == {"x": 0, "y": 1}


def test_follow_up_rows_and_columns_uses_saved_summary() -> None:
    df = _iris_like_df()
    saved_analysis = {
        "summary_for_user": "Dataset has 10,000 rows and 14 columns."
    }

    answer = answer_follow_up_question(
        df,
        "Dataset นี้มีกี่ rows และกี่ columns?",
        saved_analysis,
    )

    assert answer == "Dataset has 10,000 rows and 14 columns."


def test_follow_up_rows_and_columns_falls_back_to_dataframe_shape() -> None:
    df = _iris_like_df()

    answer = answer_follow_up_question(
        df,
        "ข้อมูลนี้มีกี่แถวและกี่คอลัมน์",
        {},
    )

    assert answer == "Dataset นี้มีทั้งหมด 6 rows และ 4 columns."


def test_follow_up_chart_request_uses_cached_dataset(monkeypatch) -> None:
    monkeypatch.setattr(charting, "_save_figure", _fake_save_figure)
    dataset_id = "cached-iris"
    DATASET_STORE[dataset_id] = _iris_like_df()
    ANALYSIS_STORE[dataset_id] = {}
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/follow-up",
            "headers": [],
            "server": ("testserver", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )

    try:
        response = asyncio.run(
            follow_up(
                FollowUpRequest(
                    dataset_id=dataset_id,
                    question="ขอสร้างกราฟ species ให้หน่อย",
                ),
                request,
            )
        )
    finally:
        DATASET_STORE.pop(dataset_id, None)
        ANALYSIS_STORE.pop(dataset_id, None)

    assert response["success"] is True
    assert response["answer"] == "สร้างกราฟจาก dataset เดิมให้แล้ว"
    assert response["chart_urls"] == ["http://testserver/test/bar.png"]
    assert response["chart_urls_text"] == "http://testserver/test/bar.png"
    assert response["used_cached_dataset"] is True
