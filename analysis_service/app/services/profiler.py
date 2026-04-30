import warnings
import re

import pandas as pd


IDENTIFIER_COLUMN_NAMES = {
    "id",
    "index",
    "rowid",
    "rowindex",
    "recordid",
    "recordindex",
}


def _normalize_column_name(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _looks_like_identifier_name(column: object) -> bool:
    raw = str(column)
    normalized = _normalize_column_name(raw)
    if normalized in IDENTIFIER_COLUMN_NAMES:
        return True
    if re.search(r"(^|[^a-zA-Z0-9])id$", raw):
        return True
    return raw.endswith("ID") or raw.endswith("Id")


def _looks_like_datetime(series: pd.Series) -> bool:
    if series.empty or series.dtype.kind in {"i", "u", "f"}:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    valid_ratio = float(parsed.notna().sum()) / float(len(series))
    return valid_ratio >= 0.6


def _looks_like_identifier_values(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) < 2:
        return False

    unique_ratio = float(non_null.nunique(dropna=True)) / float(len(non_null))
    if unique_ratio < 0.95:
        return False

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.notna().all():
        diffs = numeric.diff().dropna()
        if not diffs.empty and (diffs == 1).all():
            return True

    return unique_ratio >= 0.98


def infer_semantic_columns(df: pd.DataFrame) -> dict[str, str]:
    semantic_columns: dict[str, str] = {}

    for col in df.columns:
        series = df[col]
        if str(col).startswith("__"):
            semantic_columns[col] = "derived_metric"
        elif _looks_like_identifier_name(col) and _looks_like_identifier_values(series):
            semantic_columns[col] = "identifier"
        elif _looks_like_datetime(series):
            semantic_columns[col] = "datetime"
        elif series.dtype.kind in {"i", "u", "f"}:
            semantic_columns[col] = "numeric_feature"
        else:
            semantic_columns[col] = "categorical"

    return semantic_columns


def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    semantic_columns = infer_semantic_columns(df)
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
        "identifier_columns": identifier_columns,
    }
