# Dify Follow-up Setup

Use `/api/analyze-binary` for the first upload. This backend returns a real JSON
object, but Dify may expose the HTTP Request node output as a binary/string
`body` when the request body is a binary file upload.

Add a Python Code Node between HTTP Request and Variable Assigner.

## Code Node Input

Set the Code Node input variable:

```text
body = HTTP Request / body
```

## Code Node Python

Copy the code from:

```text
dify_code_nodes/parse_analyze_binary_response.py
```

The Code Node returns these stable string outputs:

```text
dataset_id
analysis_result
last_chart_urls
summary_for_user
```

## Variable Assigner

Assign conversation variables from the Code Node, not directly from the HTTP
Request node:

```text
conversation.dataset_id        <- Code Node / dataset_id
conversation.analysis_result   <- Code Node / analysis_result
conversation.last_chart_urls   <- Code Node / last_chart_urls
conversation.summary_for_user  <- Code Node / summary_for_user
```

## Follow-up HTTP Request

For follow-up questions, call:

```text
POST /api/follow-up
Content-Type: application/json
```

Example body:

```json
{
  "dataset_id": "{{conversation.dataset_id}}",
  "question": "{{sys.query}}"
}
```

Do not upload the file again for follow-up questions. The backend uses the
cached dataframe identified by `dataset_id`.
