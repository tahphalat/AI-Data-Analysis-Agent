import warnings
import re
from typing import Any

import pandas as pd


IDENTIFIER_EXACT_NAMES = {
    "id",
    "index",
    "rowid",
    "rowindex",
    "recordid",
    "recordindex",
    "unnamed0",
}

IDENTIFIER_NAME_TERMS = {
    "id",
    "uuid",
    "guid",
    "key",
    "code",
    "sku",
    "serial",
    "invoice",
    "transaction",
    "record",
    "row",
    "no",
    "number",
}

MEASURE_NAME_TERMS = {
    "amount",
    "price",
    "revenue",
    "sales",
    "quantity",
    "qty",
    "unit",
    "cost",
    "profit",
    "margin",
    "total",
    "subtotal",
    "length",
    "width",
    "height",
    "weight",
    "score",
    "rate",
    "ratio",
    "percent",
    "percentage",
    "temperature",
    "temp",
    "age",
    "count",
}

MEASURE_TYPE_TERMS = {
    "amount",
    "price",
    "revenue",
    "sales",
    "quantity",
    "qty",
    "unit",
    "cost",
    "profit",
    "margin",
    "total",
    "subtotal",
    "count",
}

TEXT_ID_PATTERN = re.compile(
    r"(^[A-Za-z]{2,}[-_ ]?\d+$)|"
    r"(^\d+[-_ ][A-Za-z]{2,}$)|"
    r"(^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$)"
)


def _normalize_column_name(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _column_tokens(column: object) -> set[str]:
    raw = str(column)
    tokens = {token.lower() for token in re.split(r"[^A-Za-z0-9]+", raw) if token}
    normalized = _normalize_column_name(raw)
    tokens.add(normalized)
    return tokens


def _identifier_name_score(column: object) -> tuple[float, list[str]]:
    raw = str(column)
    normalized = _normalize_column_name(raw)
    tokens = _column_tokens(column)
    score = 0.0
    reasons: list[str] = []

    if normalized in IDENTIFIER_EXACT_NAMES:
        score += 0.45
        reasons.append("name is a common identifier/index name")
    elif tokens & IDENTIFIER_NAME_TERMS:
        score += 0.25
        reasons.append("name contains identifier-like terms")

    if raw.endswith("ID") or raw.endswith("Id") or re.search(r"(^|[^A-Za-z0-9])id$", raw):
        score += 0.2
        reasons.append("name ends with id")

    if normalized.startswith("unnamed") and any(ch.isdigit() for ch in normalized):
        score += 0.35
        reasons.append("name looks like an exported row index")

    return min(score, 0.7), reasons


def _measure_name_score(column: object) -> tuple[float, list[str]]:
    normalized = _normalize_column_name(column)
    tokens = _column_tokens(column)
    score = 0.0
    reasons: list[str] = []

    if tokens & MEASURE_TYPE_TERMS:
        score += 0.45
        reasons.append("name contains measure-like terms")

    for term in MEASURE_TYPE_TERMS:
        if term in normalized and len(normalized) > len(term):
            score += 0.25
            reasons.append("name includes a measure keyword")
            break

    return min(score, 0.7), reasons


def _looks_like_datetime(series: pd.Series) -> bool:
    if series.empty or series.dtype.kind in {"i", "u", "f"}:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    valid_ratio = float(parsed.notna().sum()) / float(len(series))
    return valid_ratio >= 0.6


def _series_stats(series: pd.Series) -> dict[str, Any]:
    non_null = series.dropna()
    total_count = int(len(series))
    non_null_count = int(len(non_null))
    unique_count = int(non_null.nunique(dropna=True)) if non_null_count else 0
    unique_ratio = float(unique_count) / float(non_null_count) if non_null_count else 0.0

    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_valid_ratio = (
        float(numeric.notna().sum()) / float(non_null_count) if non_null_count else 0.0
    )
    numeric_values = numeric.dropna()
    monotonic_increasing = bool(numeric_values.is_monotonic_increasing) if len(numeric_values) >= 2 else False
    step_one = False
    if len(numeric_values) >= 2:
        diffs = numeric_values.diff().dropna()
        step_one = bool((diffs == 1).all()) if not diffs.empty else False

    text_values = non_null.astype(str)
    average_text_length = float(text_values.str.len().mean()) if non_null_count else 0.0
    code_pattern_ratio = (
        float(text_values.map(lambda value: bool(TEXT_ID_PATTERN.match(value))).sum())
        / float(non_null_count)
        if non_null_count
        else 0.0
    )

    return {
        "total_count": total_count,
        "non_null_count": non_null_count,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "numeric_valid_ratio": numeric_valid_ratio,
        "monotonic_increasing": monotonic_increasing,
        "step_one": step_one,
        "average_text_length": average_text_length,
        "code_pattern_ratio": code_pattern_ratio,
    }


def _identifier_value_score(stats: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    unique_ratio = float(stats["unique_ratio"])
    if unique_ratio >= 0.98:
        score += 0.3
        reasons.append("values are nearly all unique")
    elif unique_ratio >= 0.9:
        score += 0.18
        reasons.append("values have high cardinality")

    if stats["step_one"]:
        score += 0.35
        reasons.append("numeric values increase by one")
    elif stats["monotonic_increasing"] and unique_ratio >= 0.9:
        score += 0.18
        reasons.append("values are monotonic and mostly unique")

    if float(stats["code_pattern_ratio"]) >= 0.8:
        score += 0.3
        reasons.append("values match identifier/code patterns")

    return min(score, 0.75), reasons


def _category_score(stats: dict[str, Any]) -> tuple[float, list[str]]:
    unique_count = int(stats["unique_count"])
    unique_ratio = float(stats["unique_ratio"])
    if unique_count <= 1:
        return 0.15, ["column has one distinct value"]
    if unique_count <= 20 and unique_ratio <= 0.8:
        return 0.45, ["values repeat across a small category set"]
    if unique_ratio <= 0.2:
        return 0.35, ["values have low cardinality"]
    return 0.0, []


def _confidence_from_margin(best_score: float, second_score: float) -> float:
    margin = max(best_score - second_score, 0.0)
    confidence = 0.45 + min(best_score, 1.0) * 0.35 + min(margin, 0.5) * 0.4
    return round(min(max(confidence, 0.0), 0.99), 2)


def _semantic_entry(column: object, series: pd.Series) -> dict[str, Any]:
    stats = _series_stats(series)
    scores = {
        "identifier": 0.0,
        "measure": 0.0,
        "numeric_feature": 0.0,
        "categorical": 0.0,
        "datetime": 0.0,
        "boolean": 0.0,
        "text": 0.0,
        "derived_metric": 0.0,
        "unknown": 0.05,
    }
    reasons_by_type: dict[str, list[str]] = {key: [] for key in scores}

    if str(column).startswith("__"):
        scores["derived_metric"] = 1.0
        reasons_by_type["derived_metric"].append("internal derived metric column")

    if _looks_like_datetime(series):
        scores["datetime"] = 0.9
        reasons_by_type["datetime"].append("most values parse as dates or timestamps")

    name_identifier_score, name_identifier_reasons = _identifier_name_score(column)
    value_identifier_score, value_identifier_reasons = _identifier_value_score(stats)
    scores["identifier"] = name_identifier_score + value_identifier_score
    reasons_by_type["identifier"].extend(name_identifier_reasons)
    reasons_by_type["identifier"].extend(value_identifier_reasons)

    measure_score, measure_reasons = _measure_name_score(column)
    if series.dtype.kind in {"i", "u", "f"} or float(stats["numeric_valid_ratio"]) >= 0.8:
        scores["measure"] = measure_score + 0.25
        scores["numeric_feature"] = 0.45
        reasons_by_type["numeric_feature"].append("values are numeric")
        if measure_reasons:
            reasons_by_type["measure"].extend(measure_reasons)
            reasons_by_type["measure"].append("values are numeric")

    category_score, category_reasons = _category_score(stats)
    scores["categorical"] = category_score
    reasons_by_type["categorical"].extend(category_reasons)

    non_null_count = int(stats["non_null_count"])
    unique_count = int(stats["unique_count"])
    if unique_count <= 2 and non_null_count > 0:
        scores["boolean"] = 0.8
        reasons_by_type["boolean"].append("column has two or fewer distinct values")

    if series.dtype.kind not in {"i", "u", "f"}:
        if scores["categorical"] == 0.0:
            scores["text"] = 0.35
            reasons_by_type["text"].append("non-numeric values are not low-cardinality categories")
        if float(stats["average_text_length"]) >= 40:
            scores["text"] += 0.3
            reasons_by_type["text"].append("values are long text")

    if scores["measure"] > 0.25:
        scores["identifier"] = max(scores["identifier"] - 0.35, 0.0)
        if reasons_by_type["identifier"]:
            reasons_by_type["identifier"].append("measure-like name lowered identifier score")
    if scores["categorical"] >= 0.35:
        scores["identifier"] = max(scores["identifier"] - 0.2, 0.0)
        if reasons_by_type["identifier"]:
            reasons_by_type["identifier"].append("repeated category-like values lowered identifier score")

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    semantic_type, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if best_score < 0.25:
        semantic_type = "unknown"
        reasons_by_type["unknown"].append("no strong semantic signal found")

    return {
        "type": semantic_type,
        "confidence": _confidence_from_margin(best_score, second_score),
        "score": round(float(best_score), 3),
        "reasons": reasons_by_type[semantic_type][:5],
        "stats": {
            "unique_count": stats["unique_count"],
            "unique_ratio": round(float(stats["unique_ratio"]), 6),
            "numeric_valid_ratio": round(float(stats["numeric_valid_ratio"]), 6),
            "monotonic_increasing": stats["monotonic_increasing"],
        },
    }


def infer_semantic_profile(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {col: _semantic_entry(col, df[col]) for col in df.columns}


def infer_semantic_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        col: profile["type"]
        for col, profile in infer_semantic_profile(df).items()
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    semantic_profile = infer_semantic_profile(df)
    semantic_columns = {
        col: profile["type"]
        for col, profile in semantic_profile.items()
    }
    identifier_columns = [
        col for col, semantic_type in semantic_columns.items()
        if semantic_type == "identifier"
    ]

    missing_summary = df.isnull().sum()
    missing_summary = {
        col: int(count)
        for col, count in missing_summary.items()
        if int(count) > 0
    }

    numeric_summary = {}
    for col in numeric_cols:
        series = df[col].dropna()
        numeric_summary[col] = {
            "count": int(series.count()),
            "mean": float(series.mean()) if not series.empty else None,
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
        }

    categorical_summary = {}
    for col in categorical_cols[:5]:
        top_values = df[col].astype(str).value_counts().head(5)
        categorical_summary[col] = top_values.to_dict()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_summary": missing_summary,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "semantic_columns": semantic_columns,
        "semantic_profile": semantic_profile,
        "identifier_columns": identifier_columns,
    }
