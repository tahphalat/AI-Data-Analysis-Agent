from __future__ import annotations

from pathlib import Path
from typing import Any
import uuid

import matplotlib.pyplot as plt
import pandas as pd

from app.services.aggregations import coerce_numeric_series


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


def _resolve_column(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    requested_name: str | None,
) -> str | None:
    if not requested_name:
        return None
    if requested_name in df.columns:
        return requested_name
    mapped = column_map.get(requested_name)
    if mapped and mapped in df.columns:
        return mapped
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
    return {
        "type": chart_type,
        "title": title,
        "group_by_column": group_col,
        "metric_column": metric_col,
        "agg": str(agg_name).lower(),
        "path": chart_path,
        "rows": rows,
    }


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
    return {
        "type": "histogram",
        "title": title,
        "column": metric_col,
        "bins": bins,
        "path": chart_path,
    }


def _generate_line(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
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
    return {
        "type": "line",
        "title": title,
        "group_by_column": group_col,
        "metric_column": metric_col,
        "agg": str(agg_name).lower(),
        "path": chart_path,
        "rows": rows,
    }


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
    raise ValueError(
        f"Unsupported chart type '{chart_type}'. Supported: bar, pie, line, histogram"
    )


def _generate_phase2_default_charts(
    df: pd.DataFrame, column_map: dict[str, Any]
) -> list[dict[str, Any]]:
    default_requests: list[dict[str, Any]] = []
    if _resolve_column(df, column_map, "revenue"):
        for group_name in ("region", "category", "product"):
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
                }
            )
    return charts


def generate_basic_charts(df: pd.DataFrame) -> list[dict]:
    charts = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        df[col].dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)

        chart_path = _save_figure(fig, "hist")
        charts.append(
            {
                "type": "histogram",
                "column": col,
                "title": f"Distribution of {col}",
                "path": chart_path,
            }
        )

    if categorical_cols:
        col = categorical_cols[0]
        value_counts = df[col].astype(str).value_counts().head(10)

        fig, ax = plt.subplots()
        value_counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Top values in {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        chart_path = _save_figure(fig, "bar")
        charts.append(
            {
                "type": "bar",
                "column": col,
                "title": f"Top values in {col}",
                "path": chart_path,
            }
        )

    return charts
