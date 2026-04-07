from pathlib import Path
import uuid
import matplotlib.pyplot as plt
import pandas as pd


CHART_DIR = Path("static/charts")
CHART_DIR.mkdir(parents=True, exist_ok=True)


def generate_basic_charts(df: pd.DataFrame) -> list[dict]:
    charts = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # 1) Histogram for the first numeric column
    if numeric_cols:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        df[col].dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)

        filename = f"{uuid.uuid4().hex}_hist.png"
        filepath = CHART_DIR / filename
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

        charts.append({
            "type": "histogram",
            "column": col,
            "title": f"Distribution of {col}",
            "path": f"/static/charts/{filename}"
        })

    # 2) Bar chart for the first categorical column
    if categorical_cols:
        col = categorical_cols[0]
        value_counts = df[col].astype(str).value_counts().head(10)

        fig, ax = plt.subplots()
        value_counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Top values in {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

        filename = f"{uuid.uuid4().hex}_bar.png"
        filepath = CHART_DIR / filename
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

        charts.append({
            "type": "bar",
            "column": col,
            "title": f"Top values in {col}",
            "path": f"/static/charts/{filename}"
        })

    return charts