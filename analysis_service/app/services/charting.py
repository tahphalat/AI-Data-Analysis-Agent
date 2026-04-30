from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import matplotlib.pyplot as plt
import pandas as pd

from app.services.aggregations import COLUMN_CANDIDATES, coerce_numeric_series
from app.services.profiler import infer_semantic_profile


BASE_DIR = Path(__file__).resolve().parents[2]
CHART_DIR = BASE_DIR / "static" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_AGG = {
    "sum": "sum",
    "mean": "mean",
    "avg": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "median": "median",
}

ROW_INDEX_COLUMN = "__row_index"
SEQUENCE_INTENTS = {"sequence", "order", "ordered", "trend", "row_order"}
FREQUENCY_INTENTS = {"frequency", "frequency_distribution", "identifier_frequency"}


def _normalize_identifier(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _looks_like_date_column(series: pd.Series) -> bool:
    parsed = pd.to_datetime(series, errors="coerce")
    if len(series) == 0:
        return False
    valid_ratio = float(parsed.notna().sum()) / float(len(series))
    return valid_ratio >= 0.6


def _find_best_date_column(df: pd.DataFrame, column_map: dict[str, Any]) -> str | None:
    mapped_date_col = _resolve_column(df, column_map, "order_date")
    if mapped_date_col and _looks_like_date_column(df[mapped_date_col]):
        return mapped_date_col

    for col in df.columns:
        lowered = str(col).lower()
        if "date" in lowered or lowered.endswith("_at") or "month" in lowered:
            if _looks_like_date_column(df[col]):
                return col
    return None


def _resolve_column(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    requested_name: str | None,
) -> str | None:
    if not requested_name:
        return None
    requested = str(requested_name).strip()
    if not requested:
        return None

    if requested in df.columns:
        return requested

    normalized_requested = _normalize_identifier(requested)
    normalized_df_columns = {
        _normalize_identifier(col): col
        for col in df.columns
    }
    if normalized_requested in normalized_df_columns:
        return normalized_df_columns[normalized_requested]

    mapped = column_map.get(requested)
    if mapped and mapped in df.columns:
        return mapped

    for key, value in column_map.items():
        if _normalize_identifier(key) == normalized_requested and value in df.columns:
            return value

    for canonical, aliases in COLUMN_CANDIDATES.items():
        normalized_aliases = {
            _normalize_identifier(canonical),
            *(_normalize_identifier(alias) for alias in aliases),
        }
        if normalized_requested in normalized_aliases:
            mapped_column = column_map.get(canonical)
            if mapped_column and mapped_column in df.columns:
                return mapped_column

    if len(normalized_requested) >= 4:
        for normalized_col, raw_col in normalized_df_columns.items():
            if (
                normalized_requested in normalized_col
                or normalized_col in normalized_requested
            ):
                return raw_col

    return None


def _save_figure(fig: plt.Figure, suffix: str) -> str:
    filename = f"{uuid.uuid4().hex}_{suffix}.png"
    fig.savefig(CHART_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    return f"/static/charts/{filename}"


def _to_json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _semantic_profile(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return infer_semantic_profile(df)


def _semantic_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        column: profile["type"]
        for column, profile in _semantic_profile(df).items()
    }


def _semantic_entry(df: pd.DataFrame, column: str | None) -> dict[str, Any]:
    if not column or column == ROW_INDEX_COLUMN:
        return {"type": "derived_metric", "confidence": 1.0, "reasons": []}
    return _semantic_profile(df).get(
        column,
        {"type": "unknown", "confidence": 0.0, "reasons": ["column was not profiled"]},
    )


def _is_identifier_column(df: pd.DataFrame, column: str | None) -> bool:
    return _semantic_entry(df, column).get("type") == "identifier"


def _low_confidence_warning(df: pd.DataFrame, column: str | None) -> list[str]:
    entry = _semantic_entry(df, column)
    confidence = float(entry.get("confidence", 0.0))
    if column and column != ROW_INDEX_COLUMN and confidence < 0.7:
        return [
            f"Semantic type for {column} is low confidence ({confidence:.2f}); chart choice may need review."
        ]
    return []


def _request_intent(request: dict[str, Any]) -> str:
    return str(request.get("intent", "")).strip().lower()


def _allows_identifier_frequency(request: dict[str, Any]) -> bool:
    if bool(request.get("allow_identifier")):
        return True
    return _request_intent(request) in FREQUENCY_INTENTS


def _with_chart_metadata(
    chart: dict[str, Any],
    selection_reason: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    chart["selection_reason"] = selection_reason
    chart["warnings"] = warnings or []
    return chart


def _build_grouped_series(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str | None,
    agg: str,
) -> pd.Series:
    normalized_agg = ALLOWED_AGG.get(agg.lower())
    if not normalized_agg:
        raise ValueError(
            f"Unsupported aggregation '{agg}'. Supported: {', '.join(ALLOWED_AGG)}"
        )

    working = df[[group_col]].copy()
    working[group_col] = working[group_col].fillna("(missing)")

    if normalized_agg == "count":
        return working.groupby(group_col).size()

    if not metric_col:
        raise ValueError("Metric column is required for aggregation other than count")

    working[metric_col] = coerce_numeric_series(df[metric_col])
    working = working.dropna(subset=[metric_col])
    if working.empty:
        return pd.Series(dtype=float)

    return working.groupby(group_col)[metric_col].agg(normalized_agg)


def _generate_grouped_bar_or_pie(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    request: dict[str, Any],
    chart_type: str,
) -> dict[str, Any]:
    requested_group = request.get("group_by") or request.get("dimension")
    metric_name = request.get("metric", "revenue")
    agg_name = request.get("agg", "sum")
    top_n = int(request.get("top_n", 10))

    group_col = _resolve_column(df, column_map, requested_group)
    metric_col = _resolve_column(df, column_map, metric_name)
    if not group_col:
        raise ValueError(f"Group column '{requested_group}' not found")
    if ALLOWED_AGG.get(str(agg_name).lower()) != "count" and not metric_col:
        raise ValueError(f"Metric column '{metric_name}' not found")

    grouped = _build_grouped_series(df, group_col, metric_col, str(agg_name))
    if grouped.empty:
        raise ValueError("No data after applying aggregation")

    grouped = grouped.sort_values(ascending=False).head(top_n)
    grouped.index = grouped.index.astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    if chart_type == "bar":
        grouped.plot(kind="bar", ax=ax)
        ax.set_xlabel(group_col)
        ax.set_ylabel(str(agg_name).lower())
    else:
        grouped.plot(kind="pie", ax=ax, autopct="%1.1f%%")
        ax.set_ylabel("")

    default_title = (
        f"{str(agg_name).upper()}({metric_col}) by {group_col}"
        if metric_col
        else f"{str(agg_name).upper()} by {group_col}"
    )
    title = request.get("title", default_title)
    ax.set_title(title)
    if chart_type == "bar":
        ax.tick_params(axis="x", labelrotation=45)

    chart_path = _save_figure(fig, chart_type)

    rows = [
        {"group": _to_json_value(label), "value": _to_json_value(value)}
        for label, value in grouped.items()
    ]
    reason = (
        f"Selected {chart_type} chart to compare {str(agg_name).lower()} values by {group_col}."
        if metric_col
        else f"Selected {chart_type} chart to show record counts by {group_col}."
    )
    warnings = [
        *_low_confidence_warning(df, group_col),
        *_low_confidence_warning(df, metric_col),
    ]
    return _with_chart_metadata({
        "type": chart_type,
        "title": title,
        "group_by_column": group_col,
        "metric_column": metric_col,
        "agg": str(agg_name).lower(),
        "path": chart_path,
        "rows": rows,
    }, reason, warnings)


def _generate_histogram(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    metric_name = request.get("column") or request.get("metric") or "revenue"
    bins = int(request.get("bins", 20))
    metric_col = _resolve_column(df, column_map, metric_name)
    if not metric_col:
        raise ValueError(f"Column '{metric_name}' not found for histogram")
    if _is_identifier_column(df, metric_col) and not _allows_identifier_frequency(request):
        sequence_request = {
            **request,
            "type": "scatter",
            "x": ROW_INDEX_COLUMN,
            "y": metric_col,
            "title": request.get("title", f"{metric_col} by Row Order"),
            "intent": "sequence",
        }
        chart = _generate_scatter(df, column_map, sequence_request)
        chart["warnings"].append(
            f"{metric_col} appears to be an identifier, so histogram was not selected by default."
        )
        chart["selection_reason"] = (
            f"Converted histogram request into a sequence scatter plot because {metric_col} is an identifier."
        )
        return chart

    values = coerce_numeric_series(df[metric_col]).dropna()
    if values.empty:
        raise ValueError(f"No numeric values available for '{metric_col}'")

    fig, ax = plt.subplots(figsize=(10, 5))
    values.plot(kind="hist", bins=bins, ax=ax)
    title = request.get("title", f"Distribution of {metric_col}")
    ax.set_title(title)
    ax.set_xlabel(metric_col)
    ax.set_ylabel("Frequency")

    chart_path = _save_figure(fig, "hist")
    warnings = _low_confidence_warning(df, metric_col)
    if _is_identifier_column(df, metric_col):
        warnings.append(
            f"{metric_col} appears to be an identifier; use this histogram only as a frequency check."
        )
    return _with_chart_metadata({
        "type": "histogram",
        "title": title,
        "column": metric_col,
        "bins": bins,
        "path": chart_path,
    }, f"Selected histogram to show the value distribution of {metric_col}.", warnings)


def _resolve_axis_series(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    requested_name: str | None,
    axis_name: str,
) -> tuple[str, pd.Series]:
    if requested_name == ROW_INDEX_COLUMN:
        return ROW_INDEX_COLUMN, pd.Series(range(len(df)), index=df.index)

    resolved = _resolve_column(df, column_map, requested_name)
    if not resolved:
        raise ValueError(f"{axis_name} column '{requested_name}' not found")
    return resolved, df[resolved]


def _generate_sequence_chart(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    request: dict[str, Any],
    chart_type: str,
) -> dict[str, Any]:
    requested_x = request.get("x") or request.get("group_by") or ROW_INDEX_COLUMN
    requested_y = request.get("y") or request.get("metric") or request.get("column")
    if not requested_y:
        raise ValueError(f"Y column is required for {chart_type} sequence chart")

    x_col, x_series = _resolve_axis_series(df, column_map, requested_x, "X")
    y_col, y_series = _resolve_axis_series(df, column_map, requested_y, "Y")
    y_values = coerce_numeric_series(y_series)

    working = pd.DataFrame({"x": x_series, "y": y_values}).dropna(subset=["y"])
    if working.empty:
        raise ValueError(f"No numeric values available for '{y_col}'")

    top_n = int(request.get("top_n", 500))
    working = working.head(top_n)

    fig, ax = plt.subplots(figsize=(11, 5))
    if chart_type == "line":
        ax.plot(working["x"], working["y"], marker="o")
    else:
        ax.scatter(working["x"], working["y"])

    title = request.get("title", f"{y_col} by Row Order" if x_col == ROW_INDEX_COLUMN else f"{y_col} by {x_col}")
    ax.set_title(title)
    ax.set_xlabel("Row Order" if x_col == ROW_INDEX_COLUMN else x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", labelrotation=45)

    chart_path = _save_figure(fig, chart_type)
    rows = [
        {"x": _to_json_value(row["x"]), "y": _to_json_value(row["y"])}
        for _, row in working.iterrows()
    ]

    warnings = _low_confidence_warning(df, y_col)
    if _is_identifier_column(df, y_col):
        warnings.append(
            f"{y_col} appears to be an identifier; this chart shows sequence/order, not a numeric distribution."
        )

    return _with_chart_metadata({
        "type": chart_type,
        "title": title,
        "x_column": x_col,
        "y_column": y_col,
        "path": chart_path,
        "rows": rows,
    }, f"Selected {chart_type} chart to show {y_col} across row order or sequence.", warnings)


def _generate_scatter(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    requested_x = request.get("x") or ROW_INDEX_COLUMN
    requested_y = request.get("y") or request.get("metric") or request.get("column")
    if not requested_y:
        raise ValueError("Y column is required for scatter chart")

    return _generate_sequence_chart(
        df,
        column_map,
        {**request, "x": requested_x, "y": requested_y},
        "scatter",
    )


def _generate_line(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    intent = _request_intent(request)
    requested_group = request.get("group_by") or request.get("x")
    if intent in SEQUENCE_INTENTS or requested_group == ROW_INDEX_COLUMN:
        sequence_request = {
            **request,
            "x": requested_group or ROW_INDEX_COLUMN,
            "y": request.get("y") or request.get("metric") or request.get("column"),
        }
        return _generate_sequence_chart(df, column_map, sequence_request, "line")

    requested_group = request.get("group_by") or request.get("x") or "order_date"
    metric_name = request.get("metric", "revenue")
    agg_name = request.get("agg", "sum")
    top_n = int(request.get("top_n", 200))

    group_col = _resolve_column(df, column_map, requested_group)
    metric_col = _resolve_column(df, column_map, metric_name)
    if not group_col:
        raise ValueError(f"Group column '{requested_group}' not found for line chart")
    if ALLOWED_AGG.get(str(agg_name).lower()) != "count" and not metric_col:
        raise ValueError(f"Metric column '{metric_name}' not found")

    working = df.copy()
    datetime_series = pd.to_datetime(working[group_col], errors="coerce")
    datetime_rows = int(datetime_series.notna().sum())

    if datetime_rows > 0:
        working[group_col] = datetime_series
        working = working.dropna(subset=[group_col])
        if working.empty:
            raise ValueError("No valid datetime values for line chart")

        normalized_agg = ALLOWED_AGG.get(str(agg_name).lower())
        if not normalized_agg:
            raise ValueError(
                f"Unsupported aggregation '{agg_name}'. Supported: {', '.join(ALLOWED_AGG)}"
            )

        if normalized_agg == "count":
            grouped = working.groupby(group_col).size()
        else:
            if not metric_col:
                raise ValueError(
                    "Metric column is required for line aggregation other than count"
                )
            working[metric_col] = coerce_numeric_series(working[metric_col])
            working = working.dropna(subset=[metric_col])
            if working.empty:
                raise ValueError("No numeric values left for line chart metric")
            grouped = working.groupby(group_col)[metric_col].agg(normalized_agg)

        grouped = grouped.dropna().sort_index().head(top_n)
        x_values = [
            str(idx.date()) if hasattr(idx, "date") else str(idx) for idx in grouped.index
        ]
    else:
        grouped = _build_grouped_series(
            working, group_col=group_col, metric_col=metric_col, agg=str(agg_name)
        )
        grouped.index = grouped.index.astype(str)
        grouped = grouped.dropna().sort_index().head(top_n)
        x_values = [str(idx) for idx in grouped.index]

    if grouped.empty:
        raise ValueError("No data after applying line aggregation")

    y_values = grouped.values
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_values, y_values, marker="o")
    title = request.get(
        "title",
        f"{str(agg_name).upper()}({metric_col}) trend by {group_col}"
        if metric_col
        else f"{str(agg_name).upper()} trend by {group_col}",
    )
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(str(agg_name).lower())
    ax.tick_params(axis="x", labelrotation=45)

    chart_path = _save_figure(fig, "line")
    rows = [
        {"x": _to_json_value(x_val), "value": _to_json_value(y_val)}
        for x_val, y_val in zip(x_values, y_values)
    ]
    warnings = [
        *_low_confidence_warning(df, group_col),
        *_low_confidence_warning(df, metric_col),
    ]
    return _with_chart_metadata({
        "type": "line",
        "title": title,
        "group_by_column": group_col,
        "metric_column": metric_col,
        "agg": str(agg_name).lower(),
        "path": chart_path,
        "rows": rows,
    }, f"Selected line chart to show {str(agg_name).lower()} trend by {group_col}.", warnings)


def _generate_chart_from_request(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    chart_type = str(request.get("type", "bar")).strip().lower()
    if chart_type in {"bar", "pie"}:
        return _generate_grouped_bar_or_pie(df, column_map, request, chart_type)
    if chart_type in {"hist", "histogram"}:
        return _generate_histogram(df, column_map, request)
    if chart_type == "line":
        return _generate_line(df, column_map, request)
    if chart_type == "scatter":
        return _generate_scatter(df, column_map, request)
    raise ValueError(
        f"Unsupported chart type '{chart_type}'. Supported: bar, pie, line, scatter, histogram"
    )


def _generate_phase2_default_charts(
    df: pd.DataFrame, column_map: dict[str, Any]
) -> list[dict[str, Any]]:
    default_requests: list[dict[str, Any]] = []
    if _resolve_column(df, column_map, "revenue"):
        for group_name in ("region", "product"):
            if _resolve_column(df, column_map, group_name):
                default_requests.append(
                    {
                        "type": "bar",
                        "group_by": group_name,
                        "metric": "revenue",
                        "agg": "sum",
                        "top_n": 10,
                        "title": f"Revenue by {group_name}",
                    }
                )

        if _resolve_column(df, column_map, "category"):
            default_requests.append(
                {
                    "type": "pie",
                    "group_by": "category",
                    "metric": "revenue",
                    "agg": "sum",
                    "top_n": 8,
                    "title": "Revenue share by category",
                }
            )

        date_col = _find_best_date_column(df, column_map)
        if date_col:
            default_requests.append(
                {
                    "type": "line",
                    "group_by": date_col,
                    "metric": "revenue",
                    "agg": "sum",
                    "top_n": 60,
                    "title": f"Revenue trend by {date_col}",
                }
            )

        default_requests.append(
            {
                "type": "histogram",
                "metric": "revenue",
                "bins": 20,
                "title": "Revenue distribution",
            }
        )

    if default_requests:
        charts: list[dict[str, Any]] = []
        for request in default_requests:
            try:
                charts.append(_generate_chart_from_request(df, column_map, request))
            except ValueError:
                continue
        if charts:
            return charts

    return generate_basic_charts(df)


def generate_charts(
    df: pd.DataFrame,
    column_map: dict[str, Any] | None = None,
    chart_requests: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    column_map = column_map or {}
    if not chart_requests:
        return _generate_phase2_default_charts(df, column_map)

    charts: list[dict[str, Any]] = []
    for index, request in enumerate(chart_requests):
        if not isinstance(request, dict):
            charts.append(
                {
                    "type": "error",
                    "request_index": index,
                    "error": "chart request must be an object",
                    "selection_reason": "Chart request could not be processed.",
                    "warnings": [],
                }
            )
            continue
        try:
            charts.append(_generate_chart_from_request(df, column_map, request))
        except ValueError as exc:
            charts.append(
                {
                    "type": "error",
                    "request_index": index,
                    "error": str(exc),
                    "request": request,
                    "selection_reason": "Chart request could not be processed.",
                    "warnings": [],
                }
            )
    return charts


def generate_basic_charts(df: pd.DataFrame) -> list[dict]:
    charts = []

    semantic_columns = _semantic_columns(df)
    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns.tolist()
        if semantic_columns.get(col) in {"numeric_feature", "measure", "derived_metric"}
    ]
    skipped_identifier_cols = [
        col for col, semantic_type in semantic_columns.items()
        if semantic_type == "identifier"
    ]
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        df[col].dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)

        chart_path = _save_figure(fig, "hist")
        charts.append(
            _with_chart_metadata({
                "type": "histogram",
                "column": col,
                "title": f"Distribution of {col}",
                "path": chart_path,
            }, f"Selected histogram because {col} is a {semantic_columns.get(col)} column.", _low_confidence_warning(df, col))
        )

    if categorical_cols:
        col = categorical_cols[0]
        value_counts = df[col].astype(str).value_counts().head(10)

        fig, ax = plt.subplots()
        value_counts.plot(kind="bar", ax=ax)
        title = f"Record Count by {col}"
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        chart_path = _save_figure(fig, "bar")
        charts.append(
            _with_chart_metadata({
                "type": "bar",
                "column": col,
                "title": title,
                "path": chart_path,
                "rows": [
                    {"group": _to_json_value(label), "value": _to_json_value(value)}
                    for label, value in value_counts.items()
                ],
            }, f"Selected bar chart to show record counts by categorical column {col}.", _low_confidence_warning(df, col))
        )

    if skipped_identifier_cols and charts:
        charts[0]["warnings"].append(
            "Identifier columns were skipped for default histogram selection: "
            + ", ".join(skipped_identifier_cols)
        )

    return charts
