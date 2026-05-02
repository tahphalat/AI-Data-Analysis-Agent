from __future__ import annotations

import re
from typing import Any

import pandas as pd


COLUMN_CANDIDATES: dict[str, list[str]] = {
    "order_id": [
        "order_id",
        "ordernumber",
        "salesordernumber",
        "invoice",
        "transactionid",
    ],
    "region": ["region", "country", "countrycode", "state", "territory", "group"],
    "category": ["category", "productcategory", "subcategory", "productsubcategory"],
    "product": ["product", "productname", "item", "sku", "productnumber"],
    "quantity": ["quantity", "orderquantity", "qty", "units", "unitssold"],
    "unit_price": [
        "unit_price",
        "unitprice",
        "price",
        "sellingprice",
        "salesprice",
    ],
    "revenue": [
        "revenue",
        "salesamount",
        "amount",
        "netsales",
        "totalrevenue",
        "total_sales",
    ],
    "cost": ["cost", "unitcost", "totalcost", "cogs"],
    "margin": ["margin", "profit", "grossprofit"],
    "order_date": [
        "order_date",
        "orderdate",
        "date",
        "datetime",
        "timestamp",
        "created_at",
        "transaction_date",
        "year_month",
    ],
}


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_candidate_column(
    normalized_map: dict[str, str], candidates: list[str]
) -> str | None:
    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        for normalized_col_name, raw_col_name in normalized_map.items():
            if normalized_candidate and normalized_candidate in normalized_col_name:
                return raw_col_name
    return None


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


def _to_json_stat_value(value: Any) -> Any:
    json_value = _to_json_value(value)
    if isinstance(json_value, float):
        return round(json_value, 6)
    return json_value


def _normalize_result_key(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return normalized or "group"


def _is_numeric_metric_column(
    df: pd.DataFrame,
    column: str,
    semantic_columns: dict[str, str],
) -> bool:
    if str(column).startswith("__"):
        return False
    if semantic_columns.get(column) == "identifier":
        return False

    series = df[column].dropna()
    if series.empty:
        return False

    numeric_series = coerce_numeric_series(series)
    valid_ratio = float(numeric_series.notna().sum()) / float(len(series))
    return valid_ratio >= 0.8


def _is_groupable_column(
    df: pd.DataFrame,
    column: str,
    semantic_columns: dict[str, str],
    max_groups: int,
) -> bool:
    if str(column).startswith("__"):
        return False
    if semantic_columns.get(column) in {"identifier", "datetime", "text"}:
        return False

    series = df[column].dropna()
    if series.empty:
        return False

    unique_count = int(series.nunique(dropna=True))
    if unique_count < 2 or unique_count > max_groups:
        return False

    if semantic_columns.get(column) in {"categorical", "boolean"}:
        return True

    return df[column].dtype.kind not in {"i", "u", "f"}


def _build_generic_grouped_aggregations(
    df: pd.DataFrame,
    semantic_columns: dict[str, str] | None = None,
    max_groups: int = 30,
    max_dimensions: int = 8,
) -> dict[str, Any]:
    semantic_columns = semantic_columns or {}
    metric_columns = [
        str(column)
        for column in df.columns
        if _is_numeric_metric_column(df, str(column), semantic_columns)
    ]
    if not metric_columns:
        return {}

    group_columns = [
        str(column)
        for column in df.columns
        if _is_groupable_column(df, str(column), semantic_columns, max_groups=max_groups)
    ][:max_dimensions]
    if not group_columns:
        return {}

    grouped_results: dict[str, Any] = {}
    for group_column in group_columns:
        working = df[[group_column, *metric_columns]].copy()
        working[group_column] = working[group_column].fillna("(missing)").astype(str)
        for metric_column in metric_columns:
            working[metric_column] = coerce_numeric_series(working[metric_column])

        rows: list[dict[str, Any]] = []
        grouped = working.groupby(group_column, dropna=False)
        for group_value, group_df in grouped:
            row: dict[str, Any] = {
                group_column: _to_json_value(group_value),
                "row_count": int(len(group_df)),
            }
            for metric_column in metric_columns:
                metric_values = group_df[metric_column].dropna()
                metric_count = int(metric_values.count())
                row[f"{metric_column}_count"] = metric_count
                row[f"{metric_column}_mean"] = (
                    _to_json_stat_value(metric_values.mean()) if metric_count else None
                )
                row[f"{metric_column}_sum"] = (
                    _to_json_stat_value(metric_values.sum()) if metric_count else None
                )
                row[f"{metric_column}_min"] = (
                    _to_json_stat_value(metric_values.min()) if metric_count else None
                )
                row[f"{metric_column}_max"] = (
                    _to_json_stat_value(metric_values.max()) if metric_count else None
                )
            rows.append(row)

        rows.sort(key=lambda item: int(item.get("row_count", 0)), reverse=True)

        top_by_mean: dict[str, dict[str, Any]] = {}
        for metric_column in metric_columns:
            mean_key = f"{metric_column}_mean"
            valid_rows = [row for row in rows if row.get(mean_key) is not None]
            if not valid_rows:
                continue
            top_row = max(valid_rows, key=lambda row: float(row[mean_key]))
            top_by_mean[metric_column] = {
                group_column: top_row.get(group_column),
                "mean": top_row.get(mean_key),
                "row_count": top_row.get("row_count"),
            }

        result_key = f"by_{_normalize_result_key(group_column)}"
        grouped_results[result_key] = {
            "group_by_column": group_column,
            "metric_columns": metric_columns,
            "rows": rows,
            "top_by_mean": top_by_mean,
            "aggregation_note": (
                "Generic grouped aggregation: numeric metrics are summarized by "
                "mean, sum, min, max, and count for each group."
            ),
        }

    return grouped_results


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    if series.dtype.kind in {"i", "u", "f"}:
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("$", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def infer_business_columns(df: pd.DataFrame) -> dict[str, str | None]:
    normalized_map = {_normalize_column_name(col): col for col in df.columns}
    inferred: dict[str, str | None] = {}

    for key, candidates in COLUMN_CANDIDATES.items():
        inferred[key] = _find_candidate_column(normalized_map, candidates)

    return inferred


def ensure_revenue_column(
    df: pd.DataFrame, column_map: dict[str, str | None]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working_df = df.copy()
    enriched_column_map: dict[str, Any] = dict(column_map)

    revenue_column = enriched_column_map.get("revenue")
    quantity_column = enriched_column_map.get("quantity")
    unit_price_column = enriched_column_map.get("unit_price")

    if revenue_column and revenue_column in working_df.columns:
        normalized_revenue_column = "__normalized_revenue"
        working_df[normalized_revenue_column] = coerce_numeric_series(
            working_df[revenue_column]
        )
        enriched_column_map["revenue_source"] = revenue_column
        enriched_column_map["revenue"] = normalized_revenue_column
        enriched_column_map["revenue_derived"] = False
        return working_df, enriched_column_map

    if (
        quantity_column
        and unit_price_column
        and quantity_column in working_df.columns
        and unit_price_column in working_df.columns
    ):
        derived_revenue_column = "__derived_revenue"
        quantity = coerce_numeric_series(working_df[quantity_column])
        unit_price = coerce_numeric_series(working_df[unit_price_column])
        working_df[derived_revenue_column] = quantity * unit_price

        enriched_column_map["revenue_source"] = "derived(quantity*unit_price)"
        enriched_column_map["revenue"] = derived_revenue_column
        enriched_column_map["revenue_derived"] = True
        return working_df, enriched_column_map

    enriched_column_map["revenue_source"] = None
    enriched_column_map["revenue_derived"] = None
    return working_df, enriched_column_map


def _build_record_for_row(
    row: pd.Series, column_map: dict[str, Any], metric_name: str, metric_column: str
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "metric": metric_name,
        "metric_column": metric_column,
        metric_name: _to_json_value(row.get(metric_column)),
    }

    for canonical_field in ("order_id", "region", "category", "product"):
        raw_col = column_map.get(canonical_field)
        if raw_col and raw_col in row.index:
            record[canonical_field] = _to_json_value(row.get(raw_col))

    return record


def compute_row_level_extremes(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    metrics = metrics or ["revenue"]
    results: dict[str, Any] = {}

    for metric_name in metrics:
        metric_column = column_map.get(metric_name)
        if not metric_column or metric_column not in df.columns:
            continue

        numeric_metric = coerce_numeric_series(df[metric_column]).dropna()
        if numeric_metric.empty:
            continue

        max_index = numeric_metric.idxmax()
        min_index = numeric_metric.idxmin()

        max_row = df.loc[max_index]
        min_row = df.loc[min_index]

        results[metric_name] = {
            "metric_column": metric_column,
            "max": {
                "row_index": _to_json_value(max_index),
                "value": _to_json_value(numeric_metric.loc[max_index]),
                "record": _build_record_for_row(
                    max_row, column_map, metric_name, metric_column
                ),
            },
            "min": {
                "row_index": _to_json_value(min_index),
                "value": _to_json_value(numeric_metric.loc[min_index]),
                "record": _build_record_for_row(
                    min_row, column_map, metric_name, metric_column
                ),
            },
        }

    return results


def compute_top_k(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    metric: str = "revenue",
    k: int = 10,
) -> dict[str, Any]:
    metric_column = column_map.get(metric)
    if not metric_column or metric_column not in df.columns:
        return {}

    working = df.copy()
    working["__metric_value"] = coerce_numeric_series(working[metric_column])
    working = working.dropna(subset=["__metric_value"])
    if working.empty:
        return {}

    top_rows = working.sort_values("__metric_value", ascending=False).head(k)
    records: list[dict[str, Any]] = []

    for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
        row_record = _build_record_for_row(row, column_map, metric, metric_column)
        row_record["rank"] = rank
        row_record["value"] = _to_json_value(row["__metric_value"])
        records.append(row_record)

    return {
        metric: {
            "metric_column": metric_column,
            "k": k,
            "rows": records,
        }
    }


def compute_grouped_aggregations(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    semantic_columns: dict[str, str] | None = None,
    top_n_products: int = 15,
) -> dict[str, Any]:
    grouped_results = _build_generic_grouped_aggregations(
        df,
        semantic_columns=semantic_columns,
    )

    revenue_column = column_map.get("revenue")
    if not revenue_column or revenue_column not in df.columns:
        return grouped_results

    dimensions = [
        ("region", "by_region", None),
        ("category", "by_category", None),
        ("product", "by_product", top_n_products),
    ]

    for canonical_dim, result_key, top_n in dimensions:
        raw_dim_col = column_map.get(canonical_dim)
        if not raw_dim_col or raw_dim_col not in df.columns:
            continue

        grouped_df = df[[raw_dim_col, revenue_column]].copy()
        grouped_df[raw_dim_col] = grouped_df[raw_dim_col].fillna("(missing)").astype(str)
        grouped_df[revenue_column] = coerce_numeric_series(grouped_df[revenue_column])
        grouped_df = grouped_df.dropna(subset=[revenue_column])
        if grouped_df.empty:
            continue

        agg = (
            grouped_df.groupby(raw_dim_col, as_index=False)[revenue_column]
            .agg(["sum", "mean", "count"])
            .reset_index()
            .sort_values("sum", ascending=False)
        )
        agg = agg.rename(
            columns={
                raw_dim_col: canonical_dim,
                "sum": "revenue_sum",
                "mean": "revenue_avg",
                "count": "row_count",
            }
        )

        if top_n:
            agg = agg.head(top_n)

        rows: list[dict[str, Any]] = []
        for _, agg_row in agg.iterrows():
            rows.append(
                {
                    canonical_dim: _to_json_value(agg_row[canonical_dim]),
                    "revenue_sum": _to_json_value(agg_row["revenue_sum"]),
                    "revenue_avg": _to_json_value(agg_row["revenue_avg"]),
                    "row_count": int(agg_row["row_count"]),
                }
            )

        grouped_results[result_key] = {
            "group_by_column": raw_dim_col,
            "metric_column": revenue_column,
            "rows": rows,
        }

    return grouped_results


def compute_generic_grouped_statistics(
    df: pd.DataFrame,
    semantic_columns: dict[str, str] | None = None,
    max_group_columns: int = 5,
    max_metric_columns: int = 12,
    max_groups_per_column: int = 25,
) -> dict[str, Any]:
    semantic_columns = semantic_columns or {}
    numeric_columns = [
        str(col)
        for col in df.select_dtypes(include="number").columns.tolist()
        if not str(col).startswith("__")
        and semantic_columns.get(str(col), "numeric_feature")
        != "identifier"
        if semantic_columns.get(str(col), "numeric_feature")
        in {"measure", "numeric_feature", "derived_metric"}
    ][:max_metric_columns]
    group_columns = [
        str(col)
        for col in df.columns.tolist()
        if semantic_columns.get(str(col)) in {"categorical", "boolean"}
    ][:max_group_columns]

    if not numeric_columns or not group_columns:
        return {}

    results: dict[str, Any] = {}

    for group_col in group_columns:
        working = df[[group_col, *numeric_columns]].copy()
        working[group_col] = working[group_col].fillna("(missing)").astype(str)
        for metric_col in numeric_columns:
            working[metric_col] = coerce_numeric_series(working[metric_col])

        grouped = working.groupby(group_col, dropna=False)
        count_series = grouped.size().rename("row_count")
        mean_df = grouped[numeric_columns].mean()
        sum_df = grouped[numeric_columns].sum()
        min_df = grouped[numeric_columns].min()
        max_df = grouped[numeric_columns].max()

        sort_metric = numeric_columns[0]
        ordered_groups = (
            mean_df[sort_metric]
            .sort_values(ascending=False, na_position="last")
            .head(max_groups_per_column)
            .index
        )

        rows: list[dict[str, Any]] = []
        for group_value in ordered_groups:
            row: dict[str, Any] = {
                group_col: _to_json_value(group_value),
                "row_count": int(count_series.loc[group_value]),
            }
            for metric_col in numeric_columns:
                row[f"{metric_col}_mean"] = _to_json_stat_value(
                    mean_df.loc[group_value, metric_col]
                )
                row[f"{metric_col}_sum"] = _to_json_stat_value(
                    sum_df.loc[group_value, metric_col]
                )
                row[f"{metric_col}_min"] = _to_json_stat_value(
                    min_df.loc[group_value, metric_col]
                )
                row[f"{metric_col}_max"] = _to_json_stat_value(
                    max_df.loc[group_value, metric_col]
                )
            rows.append(row)

        top_mean_by_metric: dict[str, Any] = {}
        for metric_col in numeric_columns:
            metric_means = mean_df[metric_col].dropna()
            if metric_means.empty:
                continue
            top_group = metric_means.idxmax()
            top_mean_by_metric[metric_col] = {
                "group_by_column": group_col,
                "group": _to_json_value(top_group),
                "mean": _to_json_stat_value(metric_means.loc[top_group]),
                "row_count": int(count_series.loc[top_group]),
            }

        result_key = f"by_{group_col}"
        results[result_key] = {
            "group_by_column": group_col,
            "metric_columns": numeric_columns,
            "rows": rows,
            "top_mean_by_metric": top_mean_by_metric,
        }

    return results
