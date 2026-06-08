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
        parts.append("Return only valid JSON. Do not wrap it in markdown code fences.")
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
                "image": ("IMAGE",),
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
                    {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 4096, "min": 1, "max": 32000, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2147483647, "step": 1},
                ),
                "batch_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 1},
                ),
                "image_detail": (["auto", "low", "high"],),
                "force_json": ("BOOLEAN", {"default": False}),
                "timeout_seconds": (
                    "INT",
                    {"default": 300, "min": 5, "max": 900, "step": 5},
                ),
                "cache_buster": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2147483647, "step": 1},
                ),
                "stream": ("BOOLEAN", {"default": True}),
                "retries": (
                    "INT",
                    {"default": 1, "min": 0, "max": 5, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "ChatProviderAPI"

    def run(
        self,
        image,
        api_key,
        endpoint,
        model,
        system_preset,
        user_prompt,
        custom_system_prompt,
        temperature,
        top_p,
        max_tokens,
        seed,
        batch_index,
        image_detail,
        force_json,
        timeout_seconds,
        cache_buster,
        stream=True,
        retries=1,
        base_url=DEFAULT_BASE_URL,
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

        image_url, image_width, image_height = _image_tensor_to_data_url_and_size(
            image,
            batch_index=batch_index,
        )
        system_prompt = _build_system_prompt(system_preset, custom_system_prompt, force_json)
        system_prompt = _apply_image_dimensions(system_prompt, image_width, image_height)
        user_prompt_sent = _build_user_prompt(
            user_prompt,
            system_prompt,
            image_width,
            image_height,
        )
        effective_max_tokens = _effective_max_tokens(
            max_tokens,
            system_preset,
            system_prompt,
            force_json,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt_sent},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": image_detail},
                    },
                ],
            }
        )

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
            "stream": bool(stream),
            "image": {
                "batch_index": int(batch_index),
                "width": image_width,
                "height": image_height,
                "detail": image_detail,
                "data_url_chars": len(image_url),
            },
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
