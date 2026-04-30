import ast
import json


def _parse_json_like(value):
    if isinstance(value, dict):
        if "body" in value:
            return _parse_json_like(value["body"])
        return value

    if isinstance(value, (bytes, bytearray)):
        return _parse_json_like(value.decode("utf-8"))

    text = str(value).strip()

    if text.startswith("b'") or text.startswith('b"'):
        try:
            return _parse_json_like(ast.literal_eval(text).decode("utf-8"))
        except Exception:
            text = text[2:-1].replace('\\"', '"')

    parsed = json.loads(text)
    if isinstance(parsed, str):
        return _parse_json_like(parsed)
    return parsed


def main(body):
    """Parse Dify HTTP Request output from /api/analyze-binary.

    Dify may expose the HTTP response body as bytes, a JSON string, a string that
    looks like b'...', an already-parsed object, or a wrapper object like:
    {"body": "...", "headers": {...}, "status_code": 200}. This Code Node
    normalizes it into stable string outputs for Variable Assigner.
    """
    try:
        data = _parse_json_like(body)
        if not isinstance(data, dict):
            raise ValueError(f"Parsed response is {type(data).__name__}, expected dict")
    except Exception as exc:
        return {
            "dataset_id": "",
            "analysis_result": "error",
            "last_chart_urls": "",
            "summary_for_user": f"Failed to parse API response: {str(exc)}",
        }

    chart_urls = data.get("chart_urls", [])
    if isinstance(chart_urls, list):
        chart_urls_text = "\n".join(str(url) for url in chart_urls)
    else:
        chart_urls_text = str(chart_urls or data.get("chart_urls_text", ""))

    return {
        "dataset_id": str(data.get("dataset_id", "")),
        "analysis_result": str(data.get("status", "")),
        "last_chart_urls": chart_urls_text,
        "summary_for_user": str(data.get("summary_for_user", "")),
    }
