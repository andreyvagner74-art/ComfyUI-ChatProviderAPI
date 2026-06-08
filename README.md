# ComfyUI-ChatProviderAPI

Custom node for sending a ComfyUI `IMAGE` to ChatProvider Google AI/Gemini and returning the model response as text.

## Node

`ChatProviderAPI / ChatProvider Google-AI Vision`

## Notes

- Models are loaded from `https://chatprovider.org/api/v1/models` and filtered to `google-ai/*`.
- System prompt presets are bundled in `ChatProviderAPI_Presets.json`.
- The image is sent as an OpenAI-compatible base64 `image_url` data URL.
- `stream` is enabled by default and streamed responses are aggregated back into a single text output.
- Each request is printed to the ComfyUI console with `resolved_system_prompt` and `user_prompt`; image bytes and API keys are masked.
- API key can be entered in the node or supplied via `CHATPROVIDER_API_KEY`.
- Default chat base URL is `https://chatprovider.org/proxy`.
- Model list is still loaded from `https://chatprovider.org/api/v1/models`.

If the model list cannot be fetched, the node falls back to common Gemini model names.
