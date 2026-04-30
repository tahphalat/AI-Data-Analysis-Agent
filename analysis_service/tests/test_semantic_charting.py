import pandas as pd

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
    assert profile["semantic_columns"]["amount"] == "numeric_feature"


def test_profile_does_not_treat_valid_as_id_name() -> None:
    df = pd.DataFrame({"valid": [1, 0, 1], "amount": [10.0, 15.0, 20.0]})

    profile = profile_dataframe(df)

    assert profile["semantic_columns"]["valid"] == "numeric_feature"


def test_default_charts_skip_id_histogram_and_count_species(monkeypatch) -> None:
    monkeypatch.setattr(charting, "_save_figure", _fake_save_figure)

    charts = charting.generate_charts(_iris_like_df(), column_map={})

    histogram = next(chart for chart in charts if chart["type"] == "histogram")
    bar = next(chart for chart in charts if chart["type"] == "bar")

    assert histogram["column"] == "SepalLengthCm"
    assert histogram["column"] != "Id"
    assert "Identifier columns were skipped" in histogram["warnings"][0]
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
    assert "histogram was not selected" in chart["warnings"][0]


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
