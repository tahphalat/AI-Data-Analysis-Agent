from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.aggregations import coerce_numeric_series


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


def check_duplicates(df: pd.DataFrame, column_map: dict[str, Any]) -> dict[str, Any]:
    total_rows = int(len(df))
    if total_rows == 0:
        return {
            "exact_duplicate_rows": 0,
            "rows_in_duplicate_groups": 0,
            "duplicate_ratio": 0.0,
            "duplicate_row_indices_sample": [],
            "duplicate_order_id_count": 0,
            "duplicate_order_rows": 0,
        }

    exact_duplicate_mask = df.duplicated(keep="first")
    duplicate_group_mask = df.duplicated(keep=False)

    exact_duplicate_rows = int(exact_duplicate_mask.sum())
    rows_in_duplicate_groups = int(duplicate_group_mask.sum())

    duplicate_indices_sample = [
        _to_json_value(idx) for idx in df.index[duplicate_group_mask].tolist()[:10]
    ]

    duplicate_order_id_count = 0
    duplicate_order_rows = 0
    order_id_column = column_map.get("order_id")
    if order_id_column and order_id_column in df.columns:
        order_series = df[order_id_column].dropna().astype(str)
        if not order_series.empty:
            order_counts = order_series.value_counts()
            duplicate_order_id_count = int((order_counts > 1).sum())
            duplicate_order_rows = int(order_counts[order_counts > 1].sum())

    return {
        "exact_duplicate_rows": exact_duplicate_rows,
        "rows_in_duplicate_groups": rows_in_duplicate_groups,
        "duplicate_ratio": round(rows_in_duplicate_groups / total_rows, 6),
        "duplicate_row_indices_sample": duplicate_indices_sample,
        "duplicate_order_id_count": duplicate_order_id_count,
        "duplicate_order_rows": duplicate_order_rows,
    }


def _find_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols: list[str] = []
    for col in df.columns:
        lowered = col.lower()
        if "date" in lowered or lowered.endswith("_at"):
            date_cols.append(col)
    return date_cols


def check_missing_and_invalid(
    df: pd.DataFrame, column_map: dict[str, Any]
) -> dict[str, Any]:
    missing_counts = df.isna().sum()
    missing_by_column = {
        col: int(count) for col, count in missing_counts.items() if int(count) > 0
    }
    total_missing_cells = int(missing_counts.sum())

    invalid_numeric_values: dict[str, int] = {}
    negative_numeric_values: dict[str, int] = {}

    for metric in ("quantity", "unit_price", "revenue"):
        col = column_map.get(metric)
        if not col or col not in df.columns:
            continue

        original = df[col]
        numeric = coerce_numeric_series(original)

        invalid_count = int((original.notna() & numeric.isna()).sum())
        negative_count = int((numeric < 0).fillna(False).sum())

        if invalid_count > 0:
            invalid_numeric_values[col] = invalid_count
        if negative_count > 0:
            negative_numeric_values[col] = negative_count

    invalid_dates: dict[str, int] = {}
    for col in _find_date_columns(df):
        parsed = pd.to_datetime(df[col], errors="coerce")
        invalid_count = int((df[col].notna() & parsed.isna()).sum())
        if invalid_count > 0:
            invalid_dates[col] = invalid_count

    return {
        "total_missing_cells": total_missing_cells,
        "missing_by_column": missing_by_column,
        "invalid_numeric_values": invalid_numeric_values,
        "negative_numeric_values": negative_numeric_values,
        "invalid_dates": invalid_dates,
    }


def _sample_formula_rows(
    df: pd.DataFrame,
    mismatch_mask: pd.Series,
    columns: list[str],
    max_samples: int = 5,
) -> list[dict[str, Any]]:
    sample_rows = df.loc[mismatch_mask, columns].head(max_samples)
    records: list[dict[str, Any]] = []
    for idx, row in sample_rows.iterrows():
        record = {"row_index": _to_json_value(idx)}
        for col in columns:
            record[col] = _to_json_value(row[col])
        records.append(record)
    return records


def check_formula_consistency(
    df: pd.DataFrame,
    column_map: dict[str, Any],
    tolerance_abs: float = 1e-6,
    tolerance_rel: float = 1e-4,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    quantity_col = column_map.get("quantity")
    unit_price_col = column_map.get("unit_price")
    revenue_col = column_map.get("revenue")

    if (
        quantity_col
        and unit_price_col
        and revenue_col
        and quantity_col in df.columns
        and unit_price_col in df.columns
        and revenue_col in df.columns
    ):
        quantity = coerce_numeric_series(df[quantity_col])
        unit_price = coerce_numeric_series(df[unit_price_col])
        revenue = coerce_numeric_series(df[revenue_col])

        expected = quantity * unit_price
        valid_mask = expected.notna() & revenue.notna()
        checked_rows = int(valid_mask.sum())

        if checked_rows > 0:
            diff = (revenue - expected).abs()
            allowed = tolerance_abs + expected.abs() * tolerance_rel
            mismatch_mask = valid_mask & (diff > allowed)
            mismatch_rows = int(mismatch_mask.sum())

            check_result = {
                "status": "ok",
                "checked_rows": checked_rows,
                "mismatch_rows": mismatch_rows,
                "mismatch_ratio": round(mismatch_rows / checked_rows, 6),
                "sample_mismatches": _sample_formula_rows(
                    df,
                    mismatch_mask,
                    [quantity_col, unit_price_col, revenue_col],
                ),
            }
        else:
            check_result = {
                "status": "not_applicable",
                "reason": "no valid numeric rows for quantity, unit_price, revenue",
            }
    else:
        check_result = {
            "status": "not_applicable",
            "reason": "required columns not found",
        }

    checks["revenue_equals_quantity_times_unit_price"] = check_result

    margin_col = column_map.get("margin")
    cost_col = column_map.get("cost")
    if (
        margin_col
        and cost_col
        and revenue_col
        and margin_col in df.columns
        and cost_col in df.columns
        and revenue_col in df.columns
    ):
        margin = coerce_numeric_series(df[margin_col])
        cost = coerce_numeric_series(df[cost_col])
        revenue = coerce_numeric_series(df[revenue_col])

        expected_margin = revenue - cost
        valid_mask = expected_margin.notna() & margin.notna()
        checked_rows = int(valid_mask.sum())

        if checked_rows > 0:
            diff = (margin - expected_margin).abs()
            allowed = tolerance_abs + expected_margin.abs() * tolerance_rel
            mismatch_mask = valid_mask & (diff > allowed)
            mismatch_rows = int(mismatch_mask.sum())

            margin_check = {
                "status": "ok",
                "checked_rows": checked_rows,
                "mismatch_rows": mismatch_rows,
                "mismatch_ratio": round(mismatch_rows / checked_rows, 6),
                "sample_mismatches": _sample_formula_rows(
                    df,
                    mismatch_mask,
                    [revenue_col, cost_col, margin_col],
                ),
            }
        else:
            margin_check = {
                "status": "not_applicable",
                "reason": "no valid numeric rows for revenue, cost, margin",
            }
    else:
        margin_check = {
            "status": "not_applicable",
            "reason": "required columns not found",
        }

    checks["margin_equals_revenue_minus_cost"] = margin_check
    return checks


def assess_data_quality(
    df: pd.DataFrame, column_map: dict[str, Any]
) -> dict[str, Any]:
    duplicates = check_duplicates(df, column_map)
    missing_and_invalid = check_missing_and_invalid(df, column_map)
    formula_checks = check_formula_consistency(df, column_map)

    total_rows = max(int(len(df)), 1)
    total_cells = max(total_rows * max(int(df.shape[1]), 1), 1)

    duplicate_ratio = float(duplicates["duplicate_ratio"])
    missing_ratio = float(missing_and_invalid["total_missing_cells"]) / float(total_cells)

    formula_ratios: list[float] = []
    for check in formula_checks.values():
        if check.get("status") == "ok":
            formula_ratios.append(float(check.get("mismatch_ratio", 0.0)))
    formula_ratio = sum(formula_ratios) / len(formula_ratios) if formula_ratios else 0.0

    quality_score = 100.0
    quality_score -= duplicate_ratio * 40.0
    quality_score -= missing_ratio * 30.0
    quality_score -= formula_ratio * 30.0
    quality_score = max(0.0, min(100.0, quality_score))

    if quality_score >= 90:
        overall_assessment = "good"
    elif quality_score >= 75:
        overall_assessment = "fair"
    else:
        overall_assessment = "needs_attention"

    return {
        "duplicates": duplicates,
        "missing_and_invalid": missing_and_invalid,
        "formula_checks": formula_checks,
        "overall_score": round(quality_score, 2),
        "overall_assessment": overall_assessment,
    }
