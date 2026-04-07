import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

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
    }