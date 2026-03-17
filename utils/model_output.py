"""Helpers to parse model outputs that may include malformed JSON."""

import json
import re
from typing import Any, Optional


def strip_code_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _extract_json_span(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _escape_invalid_backslashes(text: str) -> str:
    return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)


def parse_json_response(raw: str) -> tuple[Optional[Any], Optional[str]]:
    """Best-effort JSON parse for LLM responses.

    Returns `(parsed_object, error_message)` where one of them is `None`.
    """
    cleaned = strip_code_fences(raw)
    candidates = [cleaned]

    span = _extract_json_span(cleaned)
    if span:
        candidates.append(span)

    expanded = []
    for candidate in candidates:
        q = _normalize_quotes(candidate)
        expanded.append(q)
        expanded.append(_remove_trailing_commas(q))
        expanded.append(_escape_invalid_backslashes(q))
        expanded.append(_escape_invalid_backslashes(_remove_trailing_commas(q)))

    seen = set()
    unique_candidates = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            unique_candidates.append(item)

    last_error = None
    for candidate in unique_candidates:
        try:
            return json.loads(candidate), None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    return None, last_error or "Unable to parse model response as JSON"
