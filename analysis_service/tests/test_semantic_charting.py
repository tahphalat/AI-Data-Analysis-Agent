import pandas as pd

from app.routes.analyze import _build_analysis_payload
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
            "PetalLengthCm": [1.4, 1.4, 1.3, 4.0, 4.5, 5.4],
            "PetalWidthCm": [0.2, 0.2, 0.2, 1.0, 1.5, 2.1],
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


def test_initial_analysis_payload_includes_rich_analysis_context(monkeypatch) -> None:
    monkeypatch.setattr(charting, "_save_figure", _fake_save_figure)

    payload = _build_analysis_payload(
        _iris_like_df(),
        public_base_url="http://testserver",
    )

    assert payload["chart_urls"]
    assert payload["chart_metadata"]
    assert payload["chart_markdown"]
    assert payload["analysis_result"]
    assert payload["analysis_result"]["summary_for_user"] == payload["summary_for_user"]
    assert payload["analysis_result"]["chart_metadata"] == payload["chart_metadata"]
    assert payload["chart_metadata"][0]["url"].startswith("http://testserver/test/")
    assert {
        "title",
        "chart_type",
        "columns",
        "description",
        "url",
    }.issubset(payload["chart_metadata"][0])
    assert payload["chart_metadata"][0]["url"] in payload["chart_markdown"]


def test_analysis_payload_includes_species_feature_means() -> None:
    payload = _build_analysis_payload(_iris_like_df())

    grouped_stats = payload["generic_grouped_statistics"]["by_Species"]
    setosa_row = next(
        row for row in grouped_stats["rows"] if row["Species"] == "Iris-setosa"
    )

    assert grouped_stats["group_by_column"] == "Species"
    assert grouped_stats["metric_columns"] == [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]
    assert setosa_row["SepalLengthCm_mean"] == 4.9
    assert setosa_row["PetalWidthCm_mean"] == 0.2
    assert grouped_stats["top_mean_by_metric"]["PetalLengthCm"] == {
        "group_by_column": "Species",
        "group": "Iris-virginica",
        "mean": 5.4,
        "row_count": 1,
    }
    assert (
        payload["analysis_result"]["generic_grouped_statistics"]
        == payload["generic_grouped_statistics"]
    )
    assert any(
        "Iris-virginica" in insight and "PetalLengthCm" in insight
        for insight in payload["insights"]
    )
