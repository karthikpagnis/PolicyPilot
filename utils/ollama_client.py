import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
OLLAMA_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "180"))
MAX_OLLAMA_RETRY_ATTEMPTS = int(os.getenv("OLLAMA_RETRY_ATTEMPTS", "5"))
BASE_RETRY_SECONDS = float(os.getenv("OLLAMA_RETRY_BASE_SECONDS", "1"))
MAX_RETRY_WAIT_SECONDS = float(os.getenv("OLLAMA_MAX_RETRY_WAIT_SECONDS", "2"))


def get_ollama_status(model: Optional[str] = None) -> dict:
    """Returns connection and model availability status for the local Ollama server."""
    target_model = model or OLLAMA_MODEL
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    request = urllib.request.Request(url, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=min(10.0, OLLAMA_REQUEST_TIMEOUT_SECONDS)) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:
        return {
            "ok": False,
            "host": OLLAMA_HOST,
            "model": target_model,
            "model_available": False,
            "error": str(exc),
        }

    try:
        parsed = json.loads(body)
        models = parsed.get("models", [])
    except json.JSONDecodeError:
        return {
            "ok": False,
            "host": OLLAMA_HOST,
            "model": target_model,
            "model_available": False,
            "error": "Invalid JSON response from Ollama /api/tags",
        }

    installed = [m.get("name", "") for m in models if isinstance(m, dict)]
    available = target_model in installed
    return {
        "ok": available,
        "host": OLLAMA_HOST,
        "model": target_model,
        "model_available": available,
        "installed_models": installed,
        "error": None if available else f"Model '{target_model}' not found. Run: ollama pull {target_model}",
    }


def _is_retryable_error(message: str) -> bool:
    lower = message.lower()
    return any(token in lower for token in ("429", "502", "503", "504", "timed out", "connection refused"))


def _next_wait_seconds(message: str, attempt: int) -> float:
    lower = message.lower()
    # Retry quickly after timeouts to keep pipeline responsive.
    if "timed out" in lower:
        return min(1.5, MAX_RETRY_WAIT_SECONDS)
    backoff = BASE_RETRY_SECONDS * (2 ** (attempt - 1))
    return min(backoff, MAX_RETRY_WAIT_SECONDS)


def _post_chat(payload: dict) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc}") from exc

    parsed = json.loads(body)
    message = parsed.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError(f"Unexpected Ollama response format: {body}")
    return content.strip()


def chat(prompt: str, model: Optional[str] = None) -> str:
    """Send a text-only prompt to Ollama (no images)."""
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for deterministic, faster responses
            "num_predict": 500,   # Limit max tokens for faster generation
            "top_p": 0.9,         # Nucleus sampling for efficiency
        }
    }

    for attempt in range(1, MAX_OLLAMA_RETRY_ATTEMPTS + 1):
        try:
            return _post_chat(payload)
        except Exception as exc:
            is_last = attempt == MAX_OLLAMA_RETRY_ATTEMPTS
            message = str(exc)
            if is_last or not _is_retryable_error(message):
                raise
            wait_seconds = _next_wait_seconds(message, attempt)
            print(f"Ollama call failed (attempt {attempt}/{MAX_OLLAMA_RETRY_ATTEMPTS}): {message}")
            print(f"Retrying in {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)


def chat_with_images(prompt: str, images_b64: list[str], model: Optional[str] = None) -> str:
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
    }

    for attempt in range(1, MAX_OLLAMA_RETRY_ATTEMPTS + 1):
        try:
            return _post_chat(payload)
        except Exception as exc:
            is_last = attempt == MAX_OLLAMA_RETRY_ATTEMPTS
            message = str(exc)
            if is_last or not _is_retryable_error(message):
                raise
            wait_seconds = _next_wait_seconds(message, attempt)
            print(f"Ollama call failed (attempt {attempt}/{MAX_OLLAMA_RETRY_ATTEMPTS}): {message}")
            print(f"Retrying in {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)
