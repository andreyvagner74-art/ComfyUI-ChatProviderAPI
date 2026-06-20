import base64
import http.client
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


DEFAULT_BASE_URL = "https://chatprovider.org/proxy"
MODELS_URL = "https://chatprovider.org/api/v1/models"
DEFAULT_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
PRESET_PATH = os.path.join(
    NODE_DIR,
    "ChatProviderAPI_Presets.json",
)

ENDPOINTS = {
    "Google-AI direct: /google-ai/v1/chat/completions": "/google-ai/v1/chat/completions",
}
DEFAULT_ENDPOINT = "Google-AI direct: /google-ai/v1/chat/completions"

FALLBACK_MODELS = [
    "google-ai/gemini-2.5-flash",
    "google-ai/gemini-2.5-flash-lite",
    "google-ai/gemini-2.5-pro",
    "google-ai/gemini-2.0-flash",
    "google-ai/gemini-2.0-flash-lite",
    "google-ai/gemini-2.0-flash-exp",
    "google-ai/gemini-1.5-flash",
    "google-ai/gemini-1.5-pro",
    "google-ai/gemini-flash-latest",
    "google-ai/gemini-pro-latest",
]


class TransientChatProviderError(RuntimeError):
    def __init__(self, message, status_code=None, retry_after=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.body = body


def _parse_error_body(body):
    try:
        parsed = json.loads(body)
    except Exception:
        return body.strip(), None

    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        return error.get("message") or body.strip(), error.get("type")
    return body.strip(), None


def _extract_retry_after(headers, message):
    retry_after = None
    try:
        value = headers.get("Retry-After")
        if value is not None:
            retry_after = int(float(value))
    except Exception:
        retry_after = None

    if retry_after is None:
        match = re.search(r"try again in\s+(\d+)\s+seconds?", message or "", re.I)
        if match:
            retry_after = int(match.group(1))

    return retry_after


def _read_response_text(response):
    try:
        return response.read().decode("utf-8", errors="replace")
    except http.client.IncompleteRead as exc:
        partial = exc.partial or b""
        print(
            "[ComfyUI-ChatProviderAPI] WARNING: incomplete HTTP response; "
            f"using partial body ({len(partial)} bytes)."
        )
        return partial.decode("utf-8", errors="replace")


def _read_sse_response_text(response):
    chunks = []
    data_events = 0
    heartbeat_events = 0

    try:
        while True:
            line = response.readline()
            if not line:
                break

            chunks.append(line)
            stripped = line.strip()
            if stripped.startswith(b":"):
                heartbeat_events += 1
                if heartbeat_events == 1 or heartbeat_events % 12 == 0:
                    print(
                        "[ComfyUI-ChatProviderAPI] Received proxy queue heartbeat "
                        f"({heartbeat_events}); waiting for model output..."
                    )
                continue

            if stripped.startswith(b"data:"):
                payload = stripped[5:].strip()
                if payload == b"[DONE]":
                    break
                if payload:
                    data_events += 1
    except http.client.IncompleteRead as exc:
        partial = exc.partial or b""
        if partial:
            chunks.append(partial)
        print(
            "[ComfyUI-ChatProviderAPI] WARNING: incomplete SSE response; "
            f"using partial body ({sum(len(c) for c in chunks)} bytes)."
        )
    except TimeoutError as exc:
        raw = b"".join(chunks).decode("utf-8", errors="replace")
        if data_events > 0:
            print(
                "[ComfyUI-ChatProviderAPI] WARNING: timed out while reading SSE; "
                "using partial model output."
            )
            return raw
        raise TransientChatProviderError(
            "Timed out while waiting for ChatProvider stream output. "
            "The server only sent queue heartbeats and no model data before timeout_seconds. "
            "Try increasing timeout_seconds, using a faster model, or retrying later."
        ) from exc

    raw = b"".join(chunks).decode("utf-8", errors="replace")
    if heartbeat_events and data_events == 0:
        raise TransientChatProviderError(
            "ChatProvider stream ended before model output. "
            "Only proxy queue heartbeats were received. Retry later or increase timeout_seconds."
        )
    return raw


def _http_json(method, url, headers=None, payload=None, timeout=30):
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request_headers = {**DEFAULT_HEADERS, **(headers or {})}

    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                raw = _read_sse_response_text(response)
            else:
                raw = _read_response_text(response)
    except urllib.error.HTTPError as exc:
        body = _read_response_text(exc)
        message, error_type = _parse_error_body(body)
        retry_after = _extract_retry_after(exc.headers, message)
        if exc.code == 429 or exc.code >= 500:
            details = f"ChatProvider HTTP {exc.code}: {message}"
            if error_type:
                details += f" ({error_type})"
            raise TransientChatProviderError(
                details,
                status_code=exc.code,
                retry_after=retry_after,
                body=body,
            ) from exc
        details = f"ChatProvider HTTP {exc.code}: {message}"
        if error_type:
            details += f" ({error_type})"
        raise RuntimeError(details) from exc
    except urllib.error.URLError as exc:
        raise TransientChatProviderError(f"ChatProvider request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TransientChatProviderError(
            "Timed out while reading ChatProvider response. "
            "Try increasing timeout_seconds or enabling stream."
        ) from exc

    if "text/event-stream" in content_type:
        return _parse_sse_response(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ChatProvider returned non-JSON response: {raw[:1000]}") from exc


def _http_json_with_retry(method, url, headers=None, payload=None, timeout=30, retries=1):
    max_attempts = max(1, int(retries) + 1)
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                print(
                    "[ComfyUI-ChatProviderAPI] Retrying ChatProvider request "
                    f"(attempt {attempt}/{max_attempts})..."
                )
            return _http_json(method, url, headers=headers, payload=payload, timeout=timeout)
        except TransientChatProviderError as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            delay = exc.retry_after if exc.retry_after is not None else min(2.0 * attempt, 5.0)
            delay = max(0.5, min(float(delay), 30.0))
            print(
                "[ComfyUI-ChatProviderAPI] Transient ChatProvider error: "
                f"{exc}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)

    raise RuntimeError(str(last_error)) from last_error


def _load_google_models():
    try:
        data = _http_json(
            "GET",
            MODELS_URL,
            timeout=5,
        )
        models = data.get("data", data)
        ids = []
        for model in models:
            if isinstance(model, dict) and model.get("service") == "google-ai":
                for model_id in model.get("models", []):
                    ids.extend(_normalize_google_model_id(model_id))
                continue

            model_id = model.get("id") if isinstance(model, dict) else str(model)
            ids.extend(_normalize_google_model_id(model_id))
        return sorted(set(ids)) or FALLBACK_MODELS
    except Exception as exc:
        print(f"[ComfyUI-ChatProviderAPI] Could not load models: {exc}")
        return FALLBACK_MODELS


def _normalize_google_model_id(model_id):
    if not isinstance(model_id, str):
        return []

    if model_id.startswith("google-ai/"):
        model_id = model_id.split("/", 1)[1]
    if model_id.startswith("models/"):
        model_id = model_id[len("models/") :]

    lowered = model_id.lower()
    if not (lowered.startswith("gemini") or lowered.startswith("gemma")):
        return []
    unsupported_fragments = (
        "embedding",
        "tts",
        "audio",
        "live",
        "image",
        "computer-use",
        "robotics",
    )
    if any(fragment in lowered for fragment in unsupported_fragments):
        return []

    return [f"google-ai/{model_id}"]


def _load_presets():
    try:
        with open(PRESET_PATH, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except Exception as exc:
        print(f"[ComfyUI-ChatProviderAPI] Could not load presets: {exc}")
        data = [{"name": "None", "prompt": ""}]

    presets = {}
    for item in data:
        name = str(item.get("name", "")).strip() or "Unnamed"
        prompt = str(item.get("prompt", ""))
        presets[name] = prompt

    if "None" not in presets:
        presets = {"None": "", **presets}
    return presets


DIMENSION_PATTERN = re.compile(
    r"width\s*[=:]\s*\d+\s+height\s*[=:]\s*\d+",
    re.IGNORECASE,
)
REVERSED_DIMENSION_PATTERN = re.compile(
    r"height\s*[=:]\s*\d+\s+width\s*[=:]\s*\d+",
    re.IGNORECASE,
)
PHOTO_STYLE_ALIASES = (
    "photo_style",
    "photography_style",
    "image_style",
    "style_photo",
)


def _image_tensor_to_data_url_and_size(image, batch_index=0):
    if hasattr(image, "detach"):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)

    if array.ndim == 4:
        index = max(0, min(int(batch_index), array.shape[0] - 1))
        array = array[index]
    if array.ndim != 3:
        raise ValueError(f"Expected IMAGE tensor with shape [B,H,W,C] or [H,W,C], got {array.shape}")

    height, width = int(array.shape[0]), int(array.shape[1])

    array = np.clip(array, 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)

    channels = array.shape[-1]
    if channels == 1:
        pil_image = Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
    elif channels == 3:
        pil_image = Image.fromarray(array, mode="RGB")
    elif channels == 4:
        pil_image = Image.fromarray(array, mode="RGBA")
    else:
        pil_image = Image.fromarray(array[:, :, :3], mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", width, height


def _apply_image_dimensions(text, width, height):
    if not text:
        return text

    text = DIMENSION_PATTERN.sub(f"width={width} height={height}", text)
    text = REVERSED_DIMENSION_PATTERN.sub(f"height={height} width={width}", text)
    return text


def _prompt_mentions_dimensions(text):
    lowered = (text or "").lower()
    return (
        bool(DIMENSION_PATTERN.search(lowered))
        or bool(REVERSED_DIMENSION_PATTERN.search(lowered))
        or ("image width" in lowered and "image height" in lowered)
        or ("width" in lowered and "height" in lowered and "bbox" in lowered)
    )


def _build_user_prompt(user_prompt, system_prompt, width, height):
    prompt = _apply_image_dimensions(user_prompt or "", width, height).strip()
    if _prompt_mentions_dimensions(system_prompt):
        dimension_line = f"width={width} height={height}"
        if dimension_line.lower() not in prompt.lower():
            prompt = f"{dimension_line}\n\n{prompt}" if prompt else dimension_line
    return prompt


def _resolve_prompt_dimensions(image_width, image_height, width, height):
    try:
        prompt_width = int(width)
    except (TypeError, ValueError):
        prompt_width = 0
    try:
        prompt_height = int(height)
    except (TypeError, ValueError):
        prompt_height = 0

    return (
        prompt_width if prompt_width > 0 else image_width,
        prompt_height if prompt_height > 0 else image_height,
    )


def _append_multi_image_context(
    prompt,
    image_1_width,
    image_1_height,
    image_2_width,
    image_2_height,
):
    if image_2_width is None or image_2_height is None:
        return prompt

    context = (
        f"Image 1 dimensions: width={image_1_width} height={image_1_height}\n"
        f"Image 2 dimensions: width={image_2_width} height={image_2_height}\n"
        "Image 1 is the first visual reference. Image 2 is the second visual reference. "
        "If the user asks for a character on a background, treat Image 1 as the character/subject reference and Image 2 as the background/context reference."
    )
    return f"{prompt}\n\n{context}" if prompt else context


def _join_url(base_url, path):
    normalized = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    if normalized.endswith("/api"):
        normalized = normalized[: -len("/api")] + "/proxy"
    return normalized + path


def _sanitize_for_debug(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower == "authorization":
                sanitized[key] = "Bearer ***"
            elif key_lower == "url" and isinstance(item, str) and item.startswith("data:image/"):
                header = item.split(",", 1)[0]
                sanitized[key] = f"{header},<base64 image omitted; chars={len(item)}>"
            elif key_lower == "data" and isinstance(item, str) and len(item) > 512:
                sanitized[key] = f"<base64 data omitted; chars={len(item)}>"
            else:
                sanitized[key] = _sanitize_for_debug(item)
        return sanitized

    if isinstance(value, list):
        return [_sanitize_for_debug(item) for item in value]

    return value


def _debug_log_request(method, url, headers, payload, timeout, prompt_debug=None):
    sanitized = {
        "method": method,
        "url": url,
        "headers": _sanitize_for_debug(headers),
        "timeout_seconds": timeout,
        "resolved_prompt": prompt_debug,
        "payload": _sanitize_for_debug(payload),
    }
    print("[ComfyUI-ChatProviderAPI] Sending ChatProvider request:")
    print(json.dumps(sanitized, ensure_ascii=False, indent=2))


def _parse_sse_response(raw):
    text_parts = []
    last_json = None
    finish_reason = None

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        last_json = event

        choices = event.get("choices") if isinstance(event, dict) else None
        if choices:
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content") or message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                text_parts.extend(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            continue

        candidates = event.get("candidates") if isinstance(event, dict) else None
        if candidates:
            if candidates[0].get("finishReason") is not None:
                finish_reason = candidates[0].get("finishReason")
            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts.extend(part.get("text", "") for part in parts if isinstance(part, dict))

    text = "".join(text_parts).strip()
    if text:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ]
        }
    if last_json:
        return last_json

    raw = raw.strip()
    non_comment_lines = [
        line for line in raw.splitlines() if line.strip() and not line.strip().startswith(":")
    ]
    if non_comment_lines:
        print(
            "[ComfyUI-ChatProviderAPI] WARNING: SSE response did not contain "
            "a complete JSON event; returning raw partial body."
        )
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "\n".join(non_comment_lines)}}
            ]
        }

    return {}


def _build_system_prompt(preset_name, custom_system_prompt, force_json):
    presets = _load_presets()
    parts = []
    preset_prompt = presets.get(preset_name, "")
    if preset_prompt.strip():
        parts.append(preset_prompt.strip())
    if custom_system_prompt.strip():
        parts.append(custom_system_prompt.strip())
    if force_json:
        parts.append(
            "Return exactly one valid JSON object or array. Do not wrap it in markdown "
            "code fences. Do not add explanations, comments, or trailing commas. Escape "
            "all double quotes inside string values."
        )
    return "\n\n".join(parts)


def _extract_text(response):
    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()

        candidates = response.get("candidates")
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "\n".join(part.get("text", "") for part in parts).strip()

        content = response.get("content")
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            ).strip()
        if isinstance(content, str):
            return content

    return json.dumps(response, ensure_ascii=False)


def _strip_markdown_code_fence(text):
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|JSON)?\s*\n?([\s\S]*?)\n?```", stripped)
    if match:
        return match.group(1).strip()
    return text


def _next_non_ws(text, start):
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _json_quote_can_close(text, quote_index):
    next_index = _next_non_ws(text, quote_index + 1)
    if next_index >= len(text):
        return True

    next_char = text[next_index]
    if next_char in ":}]":
        return True
    if next_char == ",":
        after_comma = _next_non_ws(text, next_index + 1)
        if after_comma >= len(text):
            return False
        value_start = text[after_comma]
        if value_start in '"{[-0123456789]}':
            return True
        return bool(re.match(r"(?:true|false|null)\b", text[after_comma:]))
    return False


def _find_balanced_json_end(text, start):
    stack = []
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return index + 1

    return None


def _json_candidate_from_start(text, start):
    if start < 0 or start >= len(text) or text[start] not in "{[":
        return None

    end = _find_balanced_json_end(text, start)
    if end is not None:
        return text[start:end].strip()

    closer = "}" if text[start] == "{" else "]"
    last = text.rfind(closer)
    if last > start:
        return text[start:last + 1].strip()
    return text[start:].strip()


def _extract_json_candidate(text):
    for match in re.finditer(r"```(?:json|JSON)?\s*([\s\S]*?)```", text):
        candidate = match.group(1).strip()
        if candidate[:1] in "{[":
            return candidate, "fence"

    stripped = _strip_markdown_code_fence(text).strip()
    if stripped[:1] in "{[":
        return _json_candidate_from_start(stripped, 0), "document"

    starts = [index for index, char in enumerate(text) if char in "{["]
    for start in starts:
        candidate = _json_candidate_from_start(text, start)
        if candidate:
            return candidate, "embedded"

    return None, None


def _looks_like_json_response(text):
    stripped = _strip_markdown_code_fence(text).strip()
    if stripped[:1] in "{[":
        return True
    return bool(re.search(r"```(?:json|JSON)?\s*[\r\n]*\s*[{[]", text))


def _strip_json_comments(text):
    result = []
    in_string = False
    escape = False
    index = 0

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
        elif char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
        elif char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and text[index:index + 2] != "*/":
                index += 1
            index = min(len(text), index + 2)
        else:
            result.append(char)
            index += 1

    return "".join(result)


def _remove_trailing_json_commas(text):
    result = []
    in_string = False
    escape = False
    index = 0

    while index < len(text):
        char = text[index]

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
        elif char == ",":
            next_index = _next_non_ws(text, index + 1)
            if next_index >= len(text) or text[next_index] not in "}]":
                result.append(char)
        else:
            result.append(char)
        index += 1

    return "".join(result)


def _escape_json_string_chars(text):
    result = []
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if not in_string:
            if char == '"':
                in_string = True
                escape = False
            result.append(char)
            continue

        if escape:
            result.append(char)
            escape = False
            continue

        if char == "\\":
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if next_char and next_char not in '"\\/bfnrtu':
                result.append("\\\\")
            else:
                result.append(char)
                escape = True
            continue

        if char == '"':
            if _json_quote_can_close(text, index):
                in_string = False
                result.append(char)
            else:
                result.append('\\"')
            continue

        if char == "\n" or char == "\r":
            result.append("\\n")
        elif char == "\t":
            result.append("\\t")
        elif ord(char) < 32:
            continue
        else:
            result.append(char)

    if in_string:
        result.append('"')
    return "".join(result)


def _quote_unquoted_json_keys(text):
    return re.sub(
        r'([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:',
        r'\1"\2":',
        text,
    )


def _replace_non_json_literals(text):
    result = []
    in_string = False
    escape = False
    index = 0
    replacements = {"None": "null", "True": "true", "False": "false"}

    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        replaced = False
        for old, new in replacements.items():
            if text.startswith(old, index):
                before = text[index - 1] if index > 0 else ""
                after_index = index + len(old)
                after = text[after_index] if after_index < len(text) else ""
                if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                    result.append(new)
                    index = after_index
                    replaced = True
                    break
        if replaced:
            continue

        result.append(char)
        index += 1

    return "".join(result)


def _close_unbalanced_json(text):
    result = []
    stack = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            result.append(char)
        elif char == "{":
            stack.append("}")
            result.append(char)
        elif char == "[":
            stack.append("]")
            result.append(char)
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
                result.append(char)
        else:
            result.append(char)

    return "".join(result) + "".join(reversed(stack))


def _normalize_json_repair_text(text):
    return (
        text.strip()
        .lstrip("\ufeff")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u201e", '"')
        .replace("\u00ab", '"')
        .replace("\u00bb", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _json_repair_attempts(candidate):
    attempts = []

    def add(value):
        value = value.strip()
        if value and value not in attempts:
            attempts.append(value)

    base = _normalize_json_repair_text(candidate)
    add(base)

    no_comments = _strip_json_comments(base)
    add(no_comments)

    no_trailing_commas = _remove_trailing_json_commas(no_comments)
    add(no_trailing_commas)

    quoted_keys = _quote_unquoted_json_keys(no_trailing_commas)
    add(quoted_keys)

    json_literals = _replace_non_json_literals(quoted_keys)
    add(json_literals)

    escaped_strings = _escape_json_string_chars(json_literals)
    add(escaped_strings)

    closed = _close_unbalanced_json(escaped_strings)
    add(closed)

    add(_remove_trailing_json_commas(closed))
    return attempts


def _parse_json_with_repair(candidate):
    last_error = None
    for attempt in _json_repair_attempts(candidate):
        try:
            return json.loads(attempt), attempt, last_error
        except json.JSONDecodeError as exc:
            last_error = exc
    return None, None, last_error


def _normalize_hex_color(value):
    if not isinstance(value, str):
        return None
    color = value.strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", color):
        return "#" + color.lstrip("#").upper()
    if re.fullmatch(r"#?[0-9a-fA-F]{3}", color):
        short = color.lstrip("#")
        return "#" + "".join(char * 2 for char in short).upper()
    return None


def _clean_color_palette(palette, limit=None):
    if isinstance(palette, dict):
        palette = palette.values()
    if not isinstance(palette, (list, tuple)):
        return []

    colors = []
    for item in palette:
        color = _normalize_hex_color(item)
        if color:
            colors.append(color)
            if limit and len(colors) >= limit:
                break
    return colors


def _normalize_bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = [max(0, min(1000, round(float(v)))) for v in value]
    except (TypeError, ValueError):
        return None
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _normalize_ideogram_caption(value):
    if not isinstance(value, dict) or "compositional_deconstruction" not in value:
        return value

    normalized = {}
    if "high_level_description" in value:
        normalized["high_level_description"] = str(value.get("high_level_description") or "")

    style = value.get("style_description")
    if isinstance(style, dict):
        normalized_style = {}
        if "aesthetics" in style:
            normalized_style["aesthetics"] = str(style.get("aesthetics") or "")
        if "lighting" in style:
            normalized_style["lighting"] = str(style.get("lighting") or "")

        photo = style.get("photo")
        if photo is None:
            for alias in PHOTO_STYLE_ALIASES:
                if style.get(alias) is not None:
                    photo = style.get(alias)
                    print(
                        "[ComfyUI-ChatProviderAPI] Renamed style_description."
                        f"{alias} to style_description.photo for Ideogram KJ import."
                    )
                    break

        medium = str(style.get("medium") or "")
        if photo is not None and ("art_style" not in style or medium.lower() == "photograph"):
            normalized_style["photo"] = str(photo)
        elif style.get("art_style") is not None:
            normalized_style["art_style"] = str(style.get("art_style") or "")
        elif photo is not None:
            normalized_style["photo"] = str(photo)

        if "medium" in style:
            normalized_style["medium"] = medium
        palette = _clean_color_palette(style.get("color_palette"), limit=16)
        if palette:
            normalized_style["color_palette"] = palette
        normalized["style_description"] = normalized_style

    decomposition = value.get("compositional_deconstruction")
    if isinstance(decomposition, dict):
        normalized_decomposition = {
            "background": str(decomposition.get("background") or ""),
            "elements": [],
        }
        elements = decomposition.get("elements")
        if isinstance(elements, list):
            for element in elements:
                if not isinstance(element, dict):
                    continue
                element_type = "text" if element.get("type") == "text" else "obj"
                normalized_element = {"type": element_type}
                bbox = _normalize_bbox(element.get("bbox"))
                if bbox:
                    normalized_element["bbox"] = bbox
                if element_type == "text":
                    normalized_element["text"] = str(element.get("text") or "")
                normalized_element["desc"] = str(element.get("desc") or "")
                palette = _clean_color_palette(element.get("color_palette"), limit=5)
                if palette:
                    normalized_element["color_palette"] = palette
                normalized_decomposition["elements"].append(normalized_element)
        normalized["compositional_deconstruction"] = normalized_decomposition

    return normalized


def _postprocess_json_response(text, expect_json=False):
    if not text:
        return text

    candidate, source = _extract_json_candidate(text)
    if not candidate:
        if expect_json:
            print("[ComfyUI-ChatProviderAPI] WARNING: expected JSON output, but no JSON block was found.")
        return _strip_markdown_code_fence(text).strip()

    should_process = expect_json or source in {"fence", "document"} or _looks_like_json_response(text)
    if not should_process:
        return text

    try:
        parsed = json.loads(candidate)
        repaired = False
    except json.JSONDecodeError:
        parsed, repaired_text, repair_error = _parse_json_with_repair(candidate)
        if parsed is None:
            if expect_json:
                details = f" at char {repair_error.pos}: {repair_error.msg}" if repair_error else ""
                raise RuntimeError(
                    "ChatProvider returned JSON-like text, but automatic JSON repair failed"
                    f"{details}. Response starts with: {candidate[:500]}"
                ) from repair_error
            return text
        candidate = repaired_text
        repaired = True

    normalized = _normalize_ideogram_caption(parsed)
    result = json.dumps(normalized, ensure_ascii=False, indent=2)
    if repaired:
        print("[ComfyUI-ChatProviderAPI] Repaired invalid JSON model output.")
    elif candidate.strip() != _strip_markdown_code_fence(text).strip():
        print("[ComfyUI-ChatProviderAPI] Extracted JSON from model output wrapper text.")
    return result


def _get_finish_reason(response):
    if not isinstance(response, dict):
        return None

    choices = response.get("choices")
    if choices:
        return choices[0].get("finish_reason")

    candidates = response.get("candidates")
    if candidates:
        return candidates[0].get("finishReason")

    return response.get("stop_reason")


def _get_usage(response):
    if not isinstance(response, dict):
        return None
    return response.get("usage") or response.get("usageMetadata")


def _get_safety_info(response):
    if not isinstance(response, dict):
        return None

    info = {}
    if response.get("promptFeedback"):
        info["promptFeedback"] = response.get("promptFeedback")

    candidates = response.get("candidates")
    if candidates:
        candidate = candidates[0]
        for key in ("finishReason", "safetyRatings", "citationMetadata"):
            if candidate.get(key) is not None:
                info[key] = candidate.get(key)

    choices = response.get("choices")
    if choices:
        choice = choices[0]
        for key in ("finish_reason", "content_filter_results", "safetyRatings"):
            if choice.get(key) is not None:
                info[key] = choice.get(key)

    return info or None


def _build_empty_response_message(response):
    diagnostics = {
        "finish_reason": _get_finish_reason(response),
        "usage": _get_usage(response),
        "safety": _get_safety_info(response),
    }
    return (
        "Gemini returned an empty response. Possible reasons: safety/censorship "
        "filter, blocked content, model refusal, unsupported model behavior, or an "
        "upstream provider issue.\n\nDiagnostics:\n"
        + json.dumps(diagnostics, ensure_ascii=False, indent=2)
    )


def _debug_log_response(response, text):
    finish_reason = _get_finish_reason(response)
    debug = {
        "finish_reason": finish_reason,
        "usage": _get_usage(response),
        "text_chars": len(text),
        "text_preview_start": text[:500],
        "text_preview_end": text[-500:] if len(text) > 500 else "",
    }
    print("[ComfyUI-ChatProviderAPI] ChatProvider response:")
    print(json.dumps(debug, ensure_ascii=False, indent=2))
    if str(finish_reason).upper() in {"MAX_TOKENS", "LENGTH"}:
        print(
            "[ComfyUI-ChatProviderAPI] WARNING: response was truncated by max_tokens. "
            "Increase max_tokens or use a shorter system prompt."
        )


def _needs_large_json_budget(system_preset, system_prompt, force_json):
    text = f"{system_preset}\n{system_prompt}".lower()
    return bool(force_json) or "json" in text or "ideogram" in text


def _effective_max_tokens(max_tokens, system_preset, system_prompt, force_json):
    requested = int(max_tokens)
    minimum = 4096 if _needs_large_json_budget(system_preset, system_prompt, force_json) else 1
    effective = max(requested, minimum)
    if effective != requested:
        print(
            "[ComfyUI-ChatProviderAPI] max_tokens raised automatically "
            f"from {requested} to {effective} for JSON/preset output."
        )
    return effective


class ChatProviderGoogleAIVision:
    @classmethod
    def INPUT_TYPES(cls):
        presets = _load_presets()
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Prompt/canvas width override. 0 = use the input image width. Convert to input to connect externally.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Prompt/canvas height override. 0 = use the input image height. Convert to input to connect externally.",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
                "endpoint": (list(ENDPOINTS.keys()),),
                "model": (_load_google_models(),),
                "system_preset": (list(presets.keys()),),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "Describe this image in detail and produce a clean prompt for image generation.",
                        "multiline": True,
                    },
                ),
                "custom_system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 32000, "step": 1},
                ),
                "retries": (
                    "INT",
                    {"default": 1, "min": 0, "max": 5, "step": 1},
                ),
                "assistant_prompt_enabled": ("BOOLEAN", {"default": False}),
                "assistant_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "ChatProviderAPI"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    def run(
        self,
        width,
        height,
        api_key,
        endpoint,
        model,
        system_preset,
        user_prompt,
        custom_system_prompt,
        temperature,
        top_p,
        max_tokens,
        retries,
        assistant_prompt_enabled=False,
        assistant_prompt="",
        timeout_seconds=60,
        seed=-1,
        batch_index=0,
        image_detail="auto",
        force_json=False,
        cache_buster=0,
        stream=False,
        base_url=DEFAULT_BASE_URL,
        image=None,
        image_2=None,
    ):
        del cache_buster

        token = (api_key or os.environ.get("CHATPROVIDER_API_KEY") or "").strip()
        if not token:
            raise ValueError("Set ChatProvider API key in the node or CHATPROVIDER_API_KEY env var.")

        if endpoint not in ENDPOINTS:
            print(
                "[ComfyUI-ChatProviderAPI] Unknown or deprecated endpoint "
                f"'{endpoint}', using {DEFAULT_ENDPOINT}."
            )
        endpoint = endpoint if endpoint in ENDPOINTS else DEFAULT_ENDPOINT
        path = ENDPOINTS[endpoint]
        model_for_request = model
        if path.startswith("/google-ai/") and model_for_request.startswith("google-ai/"):
            model_for_request = model_for_request.split("/", 1)[1]

        image_url = None
        image_width = None
        image_height = None
        if image is not None:
            image_url, image_width, image_height = _image_tensor_to_data_url_and_size(
                image,
                batch_index=batch_index,
            )
        image_2_url = None
        image_2_width = None
        image_2_height = None
        if image_2 is not None:
            image_2_url, image_2_width, image_2_height = _image_tensor_to_data_url_and_size(
                image_2,
                batch_index=batch_index,
            )

        dimension_source_width = image_2_width or image_width or 1024
        dimension_source_height = image_2_height or image_height or 1024
        prompt_width, prompt_height = _resolve_prompt_dimensions(
            dimension_source_width,
            dimension_source_height,
            width,
            height,
        )
        system_prompt = _build_system_prompt(system_preset, custom_system_prompt, force_json)
        system_prompt = _apply_image_dimensions(system_prompt, prompt_width, prompt_height)
        user_prompt_sent = _build_user_prompt(
            user_prompt,
            system_prompt,
            prompt_width,
            prompt_height,
        )
        if image_url is not None and image_2_url is not None:
            user_prompt_sent = _append_multi_image_context(
                user_prompt_sent,
                image_width,
                image_height,
                image_2_width,
                image_2_height,
            )
        effective_max_tokens = _effective_max_tokens(
            max_tokens,
            system_preset,
            system_prompt,
            force_json,
        )

        image_content = []
        if user_prompt_sent:
            image_content.append({"type": "text", "text": user_prompt_sent})
        if image_url is not None and image_2_url is not None:
            image_content.append({
                "type": "text",
                "text": "Image 1 (first reference; often subject/character):",
            })
        if image_url is not None:
            image_content.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": image_detail},
            })
        if image_2_url is not None:
            image_content.extend([
                {
                    "type": "text",
                    "text": "Image 2 (second reference; often background/context):" if image_url is not None else "Image 1 (visual reference):",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_2_url, "detail": image_detail},
                },
            ])
        if not image_content:
            image_content.append({"type": "text", "text": "Generate the requested prompt."})

        assistant_prompt_sent = (assistant_prompt or "").strip() if assistant_prompt_enabled else ""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": image_content,
            }
        )
        if assistant_prompt_sent:
            messages.append({"role": "assistant", "content": assistant_prompt_sent})

        payload = {
            "model": model_for_request,
            "messages": messages,
            "stream": bool(stream),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": effective_max_tokens,
        }
        if int(seed) >= 0:
            payload["seed"] = int(seed)

        request_url = _join_url(DEFAULT_BASE_URL, path)
        request_headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "text/event-stream" if bool(stream) else "application/json",
            "Content-Type": "application/json",
            "Origin": "https://chatprovider.org",
            "Referer": "https://chatprovider.org/",
        }
        request_timeout = int(timeout_seconds)
        prompt_debug = {
            "endpoint": endpoint,
            "selected_model": model,
            "model_sent": model_for_request,
            "system_preset": system_preset,
            "resolved_system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "user_prompt_sent": user_prompt_sent,
            "assistant_prompt_enabled": bool(assistant_prompt_enabled),
            "assistant_prompt_sent": assistant_prompt_sent,
            "stream": bool(stream),
            "image": {
                "batch_index": int(batch_index),
                "width": image_width,
                "height": image_height,
                "prompt_width": prompt_width,
                "prompt_height": prompt_height,
                "detail": image_detail,
                "data_url_chars": len(image_url) if image_url is not None else 0,
            } if image_url is not None else None,
            "image_2": None if image_2_url is None else {
                "batch_index": int(batch_index),
                "width": image_2_width,
                "height": image_2_height,
                "detail": image_detail,
                "data_url_chars": len(image_2_url),
            },
            "image_count": int(image_url is not None) + int(image_2_url is not None),
            "generation": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "requested_max_tokens": int(max_tokens),
                "effective_max_tokens": effective_max_tokens,
                "seed": int(seed),
                "force_json": bool(force_json),
                "retries": int(retries),
            },
        }

        _debug_log_request(
            "POST",
            request_url,
            request_headers,
            payload,
            request_timeout,
            prompt_debug=prompt_debug,
        )

        response = _http_json_with_retry(
            "POST",
            request_url,
            headers=request_headers,
            payload=payload,
            timeout=request_timeout,
            retries=int(retries),
        )

        text = _strip_markdown_code_fence(_extract_text(response))
        text = _postprocess_json_response(
            text,
            expect_json=_needs_large_json_budget(system_preset, system_prompt, force_json),
        )
        _debug_log_response(response, text)
        if not text.strip():
            text = _build_empty_response_message(response)
            print("[ComfyUI-ChatProviderAPI] WARNING: Gemini returned an empty response.")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "ChatProviderGoogleAIVision": ChatProviderGoogleAIVision,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatProviderGoogleAIVision": "ChatProvider Google-AI Vision",
}
