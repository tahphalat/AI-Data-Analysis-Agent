from io import BytesIO
import pandas as pd


def load_dataframe(filename: str, file_bytes: bytes) -> pd.DataFrame:
    lower_name = filename.lower()

    if lower_name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))

    if lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        return pd.read_excel(BytesIO(file_bytes))

    raise ValueError("Unsupported file format. Please upload CSV or Excel.")