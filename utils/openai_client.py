import base64
import inspect
import mimetypes
import os
from dataclasses import dataclass
from typing import Optional

import httpx
from openai import OpenAI


@dataclass(frozen=True)
class OpenAIModelConfig:
    model: str
    temperature: float
    top_p: Optional[float]
    max_tokens: int

    @classmethod
    def from_env(cls):
        model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-11-20")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        top_p_env = os.getenv("OPENAI_TOP_P")
        top_p = float(top_p_env) if top_p_env else None
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1024"))
        return cls(model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


_CLIENT = None


def _require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it before running.")


def _build_http_client(timeout_value):
    client_kwargs = {}
    if timeout_value:
        try:
            client_kwargs["timeout"] = float(timeout_value)
        except ValueError:
            pass
    proxy_url = (
        os.getenv("OPENAI_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or os.getenv("ALL_PROXY")
    )
    if proxy_url:
        params = inspect.signature(httpx.Client).parameters
        if "proxy" in params:
            client_kwargs["proxy"] = proxy_url
        elif "proxies" in params:
            client_kwargs["proxies"] = proxy_url
    return httpx.Client(**client_kwargs)


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _require_api_key()
        client_kwargs = {}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            base_url = base_url.rstrip("/")
            if base_url.endswith("/chat/completions"):
                base_url = base_url[: -len("/chat/completions")]
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            client_kwargs["base_url"] = base_url
        organization = os.getenv("OPENAI_ORG")
        if organization:
            client_kwargs["organization"] = organization
        project = os.getenv("OPENAI_PROJECT")
        if project:
            client_kwargs["project"] = project
        timeout = os.getenv("OPENAI_TIMEOUT")
        client_kwargs["http_client"] = _build_http_client(timeout)
        _CLIENT = OpenAI(**client_kwargs)
    return _CLIENT


def _image_to_data_url(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_message_parts(parts):
    normalized = []
    for part in parts:
        part_type = part.get("type")
        if part_type == "text":
            normalized.append({"type": "text", "text": part["text"]})
        elif part_type == "image":
            normalized.append({
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(part["path"])},
            })
        elif part_type == "image_url":
            normalized.append(part)
        else:
            raise ValueError(f"Unsupported message part type: {part_type}")
    return normalized


def _chat_completion(content, config: OpenAIModelConfig) -> str:
    client = _get_client()
    request_kwargs = {
        "model": config.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.top_p is not None:
        request_kwargs["top_p"] = config.top_p
    response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content.strip()


def get_openai_response_with_parts(parts, config: OpenAIModelConfig) -> str:
    content = _normalize_message_parts(parts)
    return _chat_completion(content, config)


def get_openai_response(prompt: str, image_path: str, config: OpenAIModelConfig) -> str:
    parts = [
        {"type": "text", "text": prompt},
        {"type": "image", "path": image_path},
    ]
    return get_openai_response_with_parts(parts, config)
