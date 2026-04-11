from __future__ import annotations

from typing import Any


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)


def _first_group_row(
    grouped_aggregations: dict[str, Any], key: str
) -> dict[str, Any] | None:
    section = grouped_aggregations.get(key, {})
    rows = section.get("rows", [])
    if rows:
        return rows[0]
    return None


def build_summary_for_user(
    profile: dict[str, Any],
    top_k: dict[str, Any],
    grouped_aggregations: dict[str, Any],
    data_quality: dict[str, Any],
    column_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    column_map = column_map or {}

    rows = int(profile.get("rows", 0))
    columns = int(profile.get("columns", 0))
    numeric_cols = len(profile.get("numeric_columns", []))
    categorical_cols = len(profile.get("categorical_columns", []))

    highlights: list[str] = [
        f"Dataset has {rows:,} rows and {columns} columns ({numeric_cols} numeric, {categorical_cols} categorical)."
    ]

    top_revenue_rows = top_k.get("revenue", {}).get("rows", [])
    if top_revenue_rows:
        top_row = top_revenue_rows[0]
        order_id = top_row.get("order_id", "N/A")
        revenue_value = _format_number(top_row.get("value"))
        highlights.append(
            f"Top order by revenue is {order_id} with revenue {revenue_value}."
        )

    top_region = _first_group_row(grouped_aggregations, "by_region")
    if top_region:
        highlights.append(
            f"Top region by revenue is {top_region.get('region', 'N/A')} "
            f"({_format_number(top_region.get('revenue_sum'))})."
        )

    top_category = _first_group_row(grouped_aggregations, "by_category")
    if top_category:
        highlights.append(
            f"Top category by revenue is {top_category.get('category', 'N/A')} "
            f"({_format_number(top_category.get('revenue_sum'))})."
        )

    top_product = _first_group_row(grouped_aggregations, "by_product")
    if top_product:
        highlights.append(
            f"Top product by revenue is {top_product.get('product', 'N/A')} "
            f"({_format_number(top_product.get('revenue_sum'))})."
        )

    duplicates = data_quality.get("duplicates", {})
    formula_checks = data_quality.get("formula_checks", {})
    quality_notes: list[str] = []

    duplicate_ratio = duplicates.get("duplicate_ratio", 0.0)
    if duplicate_ratio and duplicate_ratio > 0:
        duplicate_rows = duplicates.get("rows_in_duplicate_groups", 0)
        quality_notes.append(
            f"Duplicate check found {duplicate_rows:,} rows in duplicate groups."
        )
    else:
        quality_notes.append("Duplicate check did not find duplicate rows.")

    revenue_formula = formula_checks.get(
        "revenue_equals_quantity_times_unit_price", {}
    )
    if revenue_formula.get("status") == "ok":
        mismatch_rows = int(revenue_formula.get("mismatch_rows", 0))
        checked_rows = int(revenue_formula.get("checked_rows", 0))
        if mismatch_rows > 0:
            quality_notes.append(
                f"Formula check found {mismatch_rows:,} mismatches out of {checked_rows:,} checked rows."
            )
        else:
            quality_notes.append(
                "Formula check passed for revenue = quantity * unit price."
            )

    overall_score = data_quality.get("overall_score")
    overall_assessment = data_quality.get("overall_assessment")
    if overall_score is not None and overall_assessment:
        quality_notes.append(
            f"Overall data quality is {overall_assessment} (score {overall_score})."
        )

    recommended_next_questions = []
    if column_map.get("revenue"):
        recommended_next_questions.extend(
            [
                "Show top 10 orders by revenue with region and category.",
                "Compare revenue distribution across regions and categories.",
            ]
        )
    if column_map.get("product"):
        recommended_next_questions.append(
            "Which products are driving most of the revenue in each region?"
        )

    return {
        "headline": "Enriched analysis completed",
        "highlights": highlights,
        "data_quality_notes": quality_notes,
        "recommended_next_questions": recommended_next_questions,
    }
