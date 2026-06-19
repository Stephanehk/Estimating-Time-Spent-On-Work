"""LLM I/O for the pairwise time-preference classifier (OpenAI Responses API).

The paper uses GPT-5.2. To obtain per-token logprobs we POST directly to the OpenAI
Responses HTTP endpoint (/v1/responses) with reasoning disabled, so the single output
token (the label 0/1/2/3) carries a logprob we can threshold on.

Only needed when (re)computing pairwise judgements. With the shipped cache present these
functions are never called. Requires OPENAI_API_KEY (optionally OPENAI_ORG_ID,
OPENAI_PROJECT_ID, OPENAI_BASE_URL) to recompute.
"""

import os
import json
import random
import time
import urllib.error
import urllib.request


# Default model for time classification (OpenAI Responses API with logprobs).
DEFAULT_MODEL = "gpt-5.2"


_BEAM_NUMERIC_TO_TEXT_LABEL = {
    "0": "TASK_1",
    "1": "TASK_2",
    "2": "EQUAL_TIME",
    "3": "CANNOT_DECIDE",
}

# OpenAI Responses API pricing: (input $ per token, output $ per token).
# Sources: OpenAI pricing pages; update when rates change.
_OPENAI_RESPONSES_PRICE_PER_TOKEN = {
    "gpt-5.2": (1.75 / 1e6, 14.0 / 1e6),
    "gpt-5.1": (1.25 / 1e6, 10.0 / 1e6),
    "gpt-5.4-mini": (0.25 / 1e6, 2.0 / 1e6),
}

_CLASSIFIER_SYSTEM_INSTRUCTION = (
    "You are a strict classifier. "
    "Output exactly one digit and nothing else: "
    "0, 1, 2, or 3."
)


def _is_retryable_openai_responses_error(exc):
    """
    Transient failures for OpenAI Responses HTTP calls (rate limits + server 5xx) and
    legacy SDK-style errors (status_code on exception objects).
    """
    error_text = str(exc)
    if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "Too Many Requests" in error_text:
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    return False


def _openai_responses_api_base_url():
    """
    Base URL for OpenAI HTTP calls (no trailing slash).
    Uses OPENAI_BASE_URL if set; otherwise https://api.openai.com/v1.
    """
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    assert base, "OPENAI_BASE_URL, if set, must be non-empty."
    return base


_MAX_OPENAI_ERROR_BODY_PRINT_CHARS = 24000


def _print_openai_responses_http_failure(url, body, exc):
    """
    Print diagnostics for failed OpenAI Responses HTTP calls.
    Does not log secrets; may log model name and payload size from body.
    """
    print("--- OpenAI Responses HTTP failure ---")
    print(f"  url: {url}")
    if isinstance(body, dict):
        model = body.get("model")
        if model is not None:
            print(f"  request.model: {model}")
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        print(f"  request JSON size (bytes): {len(payload)}")
    if isinstance(exc, urllib.error.HTTPError):
        print(f"  HTTP status: {exc.code} {getattr(exc, 'reason', '')!s}".rstrip())
        hdrs = exc.headers
        if hdrs:
            for name in ("x-request-id", "openai-processing-ms", "cf-ray", "date", "retry-after"):
                val = hdrs.get(name) or hdrs.get(name.title())
                if val:
                    print(f"  response header {name}: {val}")
        raw = b""
        try:
            raw = exc.read()
        except Exception as read_exc:
            print(f"  (could not read error response body: {read_exc!r})")
        else:
            text = raw.decode("utf-8", errors="replace")
            if len(text) > _MAX_OPENAI_ERROR_BODY_PRINT_CHARS:
                text = text[:_MAX_OPENAI_ERROR_BODY_PRINT_CHARS] + "\n... [truncated] ..."
            print(f"  response body ({len(raw)} bytes):")
            print(text)
            if raw:
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, dict) and parsed.get("error") is not None:
                        print(f"  parsed error object: {parsed['error']!r}")
    else:
        print(f"  exception type: {type(exc).__name__}")
        print(f"  exception: {exc!r}")
    print("--- end OpenAI Responses HTTP failure ---")


def _openai_responses_http_post(body, timeout_sec):
    """
    POST JSON to POST /v1/responses. Returns parsed JSON object (dict).
    Assumes OPENAI_API_KEY is set; optional OPENAI_ORG_ID, OPENAI_PROJECT_ID as headers.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set for OpenAI HTTP calls."
    url = f"{_openai_responses_api_base_url()}/responses"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    organization = os.environ.get("OPENAI_ORG_ID")
    if organization:
        req.add_header("OpenAI-Organization", organization)
    project = os.environ.get("OPENAI_PROJECT_ID")
    if project:
        req.add_header("OpenAI-Project", project)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        _print_openai_responses_http_failure(url, body, exc)
        raise
    except urllib.error.URLError as exc:
        print("--- OpenAI Responses HTTP failure (network) ---")
        print(f"  url: {url}")
        print(f"  URLError: {exc!r}")
        if getattr(exc, "reason", None) is not None:
            print(f"  reason: {exc.reason!r}")
        print("--- end ---")
        raise
    return json.loads(raw)


def _assert_openai_responses_completed(resp_json):
    assert isinstance(resp_json, dict)
    err = resp_json.get("error")
    assert err is None, f"OpenAI Responses API error: {err}"
    st = resp_json.get("status")
    assert st == "completed", f"OpenAI Responses API status not completed: {st!r}"


def _openai_responses_cost_from_usage(usage, model_name):
    """
    Compute cost in USD from Responses API usage (input_tokens, output_tokens).
    usage may be an object or dict with input_tokens and output_tokens.
    """
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
    else:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
    assert float(input_tokens) >= 0.0, f"Unexpected input token count: {input_tokens!r}"
    assert float(output_tokens) >= 0.0, f"Unexpected output token count: {output_tokens!r}"
    prices = _OPENAI_RESPONSES_PRICE_PER_TOKEN.get(model_name)
    assert prices is not None, f"No pricing for model {model_name}; add to _OPENAI_RESPONSES_PRICE_PER_TOKEN."
    price_in, price_out = prices
    return float(input_tokens) * price_in + float(output_tokens) * price_out


def _openai_responses_label_with_logprob(prompt, model_name):
    """
    OpenAI Responses API (HTTP) for label + logprob (GPT-5 logprobs via include).
    Returns (text_label, label_logprob, cost). Cost is computed from response usage (input/output tokens).
    """
    body = {
        "model": model_name,
        "input": [
            {"role": "system", "content": _CLASSIFIER_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "include": ["message.output_text.logprobs"],
    }
    max_attempts = 8
    initial_backoff_seconds = 2.0
    backoff_multiplier = 2.0
    max_backoff_seconds = 60.0
    backoff_seconds = initial_backoff_seconds
    for attempt in range(1, max_attempts + 1):
        try:
            resp_json = _openai_responses_http_post(body, timeout_sec=240)
            break
        except Exception as exc:
            if not _is_retryable_openai_responses_error(exc):
                raise
            if attempt == max_attempts:
                raise
            jitter_seconds = random.uniform(0.0, 0.75)
            sleep_seconds = min(backoff_seconds + jitter_seconds, max_backoff_seconds)
            print(
                f"OpenAI transient error (attempt {attempt}/{max_attempts}): {exc!r}; "
                f"sleeping {sleep_seconds:.2f}s before retry."
            )
            time.sleep(sleep_seconds)
            backoff_seconds = min(backoff_seconds * backoff_multiplier, max_backoff_seconds)
    else:
        raise RuntimeError("Unexpected retry loop termination in _openai_responses_label_with_logprob.")

    _assert_openai_responses_completed(resp_json)
    output = resp_json.get("output")
    assert output, "OpenAI Responses API response has no output."
    msg = next((item for item in output if item.get("type") == "message"), None)
    assert msg is not None, "No message item in Responses API output."
    content = msg.get("content") or []
    assert content, "Message has no content."
    text_block = content[0]
    if isinstance(text_block, dict):
        response_text = (text_block.get("text") or "").strip()
        logprobs_list = text_block.get("logprobs") or []
    else:
        response_text = (getattr(text_block, "text", None) or "").strip()
        logprobs_list = getattr(text_block, "logprobs", None) or []
    assert response_text in _BEAM_NUMERIC_TO_TEXT_LABEL, f"Invalid classifier output: {response_text}"

    assert len(logprobs_list) > 0, "Expected at least one token logprob from Responses API."
    assert len(logprobs_list) == 1, "Expected exactly one token logprob from Responses API."
    label_logprob = 0.0
    for entry in logprobs_list:
        lp = entry.get("logprob") if isinstance(entry, dict) else getattr(entry, "logprob", None)
        assert lp is not None, "Logprob entry missing logprob."
        label_logprob += float(lp)

    text_label = _BEAM_NUMERIC_TO_TEXT_LABEL[response_text]
    usage = resp_json.get("usage")
    assert usage is not None, "OpenAI Responses API response missing usage."
    cost = _openai_responses_cost_from_usage(usage, model_name)
    return text_label, label_logprob, cost


def send_prompt_for_label_with_logprob(prompt, model_name=DEFAULT_MODEL):
    """
    Sends a prompt that must return exactly one numeric label token among 0, 1, 2, 3.
    Returns (text_label, logprob_of_output_label, cost), where text_label is one of
    TASK_1, TASK_2, EQUAL_TIME, CANNOT_DECIDE.

    Uses HTTP POST to the OpenAI Responses API so logprobs are available; thinking is
    disabled via reasoning={"effort": "none"}.
    """
    return _openai_responses_label_with_logprob(prompt, model_name)
