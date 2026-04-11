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
    top_n_products: int = 15,
) -> dict[str, Any]:
    revenue_column = column_map.get("revenue")
    if not revenue_column or revenue_column not in df.columns:
        return {}

    dimensions = [
        ("region", "by_region", None),
        ("category", "by_category", None),
        ("product", "by_product", top_n_products),
    ]

    grouped_results: dict[str, Any] = {}

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
