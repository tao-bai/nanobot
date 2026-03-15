"""Gemini Native provider — uses google-genai SDK directly for custom endpoints."""

from __future__ import annotations

import json
import re
import secrets
import string
from typing import Any

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALNUM = string.ascii_letters + string.digits

_DATA_URI_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


def _parse_data_uri(data_uri: str) -> tuple[str, str] | None:
    """Parse a data URI into (mime_type, base64_data) or None if invalid."""
    m = _DATA_URI_RE.match(data_uri)
    return (m.group(1), m.group(2)) if m else None


def _short_tool_id() -> str:
    """Generate a 9-char alphanumeric ID compatible with all providers."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class GeminiNativeProvider(LLMProvider):
    """LLM provider using google-genai SDK directly for custom Gemini endpoints."""

    # Proxy endpoints often need more retries for transient capacity issues.
    _CHAT_RETRY_DELAYS = (1, 2, 4, 8, 16)
    _TRANSIENT_ERROR_MARKERS = LLMProvider._TRANSIENT_ERROR_MARKERS + (
        "accounts exhausted",
        "capacity",
    )

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "http://localhost:8045",
        default_model: str = "gemini-3-flash",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers

        # Lazy-init client on first use to avoid import at module level
        self._client = None
        # Maps tool_call_id -> function name for tool result conversion
        self._tool_call_name_map: dict[str, str] = {}

    def _get_client(self):
        """Lazily create the google-genai Client."""
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as e:
                raise ImportError(
                    "google-genai is required for Gemini native provider. "
                    "Install it with: pip install google-genai"
                ) from e

            http_opts: dict[str, Any] = {}
            if self.api_base:
                http_opts["base_url"] = self.api_base
            if self.extra_headers:
                http_opts["headers"] = self.extra_headers

            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(**http_opts) if http_opts else None,
            )
        return self._client

    def _strip_model_prefix(self, model: str) -> str:
        """Strip gemini/ prefix — the SDK expects bare model names."""
        if model.startswith("gemini/"):
            return model[len("gemini/"):]
        return model

    @staticmethod
    def _convert_content_to_parts(content: Any) -> list[dict[str, Any]]:
        """Convert OpenAI-format content (str, list, or None) to Gemini-format parts."""
        if isinstance(content, str):
            return [{"text": content}]
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type", "")
                if item_type == "text":
                    parts.append({"text": item.get("text", "")})
                elif item_type == "image_url":
                    url_info = item.get("image_url", {})
                    url = url_info.get("url", "") if isinstance(url_info, dict) else ""
                    parsed = _parse_data_uri(url)
                    if parsed:
                        mime_type, b64_data = parsed
                        parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
                    else:
                        # Non-data-URI: text fallback
                        parts.append({"text": f"[image: {url}]"})
            return parts if parts else [{"text": ""}]
        return [{"text": content or ""}]

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert OpenAI-format messages to Gemini SDK content dicts.

        Returns:
            (system_instruction, contents) where contents is a list of
            Gemini-format content dicts with 'role' and 'parts'.
        """
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            if role == "system":
                if content:
                    system_parts.append(content)
                continue

            if role == "user":
                contents.append({
                    "role": "user",
                    "parts": self._convert_content_to_parts(content),
                })
                continue

            if role == "assistant":
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        # Alias names so history matches what _convert_tools declared
                        name = self._TOOL_NAME_ALIASES.get(name, name)
                        args_raw = func.get("arguments", "{}")
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        parts.append({"function_call": {"name": name, "args": args}})
                        # Populate name map so subsequent tool results can resolve names
                        tc_id = tc.get("id", "")
                        if tc_id:
                            self._tool_call_name_map[tc_id] = name
                    # Also include text content if present alongside tool calls
                    if content:
                        parts.insert(0, {"text": content})
                    contents.append({"role": "model", "parts": parts})
                else:
                    contents.append({
                        "role": "model",
                        "parts": [{"text": content or ""}],
                    })
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                func_name = self._tool_call_name_map.get(tool_call_id, "unknown")
                # Parse content as JSON if possible, else wrap in dict
                try:
                    response_data = json.loads(content) if isinstance(content, str) else content
                    if not isinstance(response_data, dict):
                        response_data = {"result": response_data}
                except (json.JSONDecodeError, TypeError):
                    response_data = {"result": content or ""}

                part = {"function_response": {"name": func_name, "response": response_data}}

                # Merge consecutive tool results into one user Content
                if contents and contents[-1].get("role") == "user" and any(
                    "function_response" in p for p in contents[-1].get("parts", [])
                ):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
                continue

        system_instruction = "\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    # Tools whose names conflict with Gemini built-in tools.  We rename them
    # before sending to the API and map back in the response.
    _TOOL_NAME_ALIASES = {"web_search": "search_web"}
    _TOOL_NAME_ALIASES_REV = {v: k for k, v in _TOOL_NAME_ALIASES.items()}

    def _convert_tools(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI tool definitions to Gemini FunctionDeclaration dicts."""
        if not tools:
            return None
        declarations = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            # Rename tools that clash with Gemini built-ins
            name = self._TOOL_NAME_ALIASES.get(name, name)
            decl: dict[str, Any] = {"name": name}
            if func.get("description"):
                decl["description"] = func["description"]
            if func.get("parameters"):
                decl["parameters_json_schema"] = func["parameters"]
            declarations.append(decl)
        return [{"function_declarations": declarations}]

    def _convert_tool_choice(
        self, tool_choice: str | dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Convert OpenAI tool_choice to Gemini FunctionCallingConfig dict."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
            mode = mode_map.get(tool_choice)
            if mode:
                return {"function_calling_config": {"mode": mode}}
            return None
        # Specific function: {"type": "function", "function": {"name": "X"}}
        if isinstance(tool_choice, dict):
            func_name = tool_choice.get("function", {}).get("name")
            if func_name:
                func_name = self._TOOL_NAME_ALIASES.get(func_name, func_name)
                return {
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [func_name],
                    }
                }
        return None

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Gemini GenerateContentResponse into LLMResponse."""
        # Extract text and function calls from first candidate
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        # Reset tool call name map for next turn
        self._tool_call_name_map = {}

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return LLMResponse(content=getattr(response, "text", None), finish_reason="stop")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        for part in parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            if getattr(part, "function_call", None):
                fc = part.function_call
                tc_id = _short_tool_id()
                name = getattr(fc, "name", "") or ""
                # Map aliased names back to the original tool name
                name = self._TOOL_NAME_ALIASES_REV.get(name, name)
                args = dict(getattr(fc, "args", {}) or {})
                tool_calls.append(ToolCallRequest(id=tc_id, name=name, arguments=args))
                self._tool_call_name_map[tc_id] = name

        finish_reason = "tool_calls" if tool_calls else "stop"
        combined_text = "\n".join(text_parts) if text_parts else None

        # Extract usage
        usage: dict[str, int] = {}
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            prompt = getattr(usage_meta, "prompt_token_count", None) or 0
            completion = getattr(usage_meta, "candidates_token_count", None) or 0
            total = getattr(usage_meta, "total_token_count", None) or (prompt + completion)
            usage = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }

        return LLMResponse(
            content=combined_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request via google-genai SDK."""
        model = self._strip_model_prefix(model or self.default_model)
        messages = self._sanitize_empty_content(messages)
        max_tokens = max(1, max_tokens)

        system_instruction, contents = self._convert_messages(messages)
        gemini_tools = self._convert_tools(tools)
        tool_config = self._convert_tool_choice(tool_choice)

        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_instruction:
            config["system_instruction"] = system_instruction
        if gemini_tools:
            config["tools"] = gemini_tools
        if tool_config:
            config["tool_config"] = tool_config
        # Disable automatic function calling — nanobot manages the tool loop
        config["automatic_function_calling"] = {"disable": True}

        try:
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return self._parse_response(response)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        return self.default_model
