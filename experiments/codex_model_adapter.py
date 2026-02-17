"""
Codex Responses API Adapter for CAMEL-AI Framework.

This module provides a custom model class that adapts CAMEL's ChatAgent
(which expects OpenAI Chat Completions API) to work with the Codex
Responses API endpoint.

The adapter intercepts chat.completions.create() calls and translates them
to the /responses endpoint format, then converts the response back to
ChatCompletion format that CAMEL expects.
"""

import os
import json
import time
import httpx
from typing import Any, Dict, List, Optional, Type, Union

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from camel.models import BaseModelBackend
from camel.types import ModelPlatformType


# ============================================================
# Fake OpenAI client that routes to Responses API
# ============================================================

class CodexResponsesClient:
    """A client that mimics OpenAI's chat.completions interface
    but actually calls the Codex Responses API."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(300.0, connect=30.0),
            verify=False,
            transport=httpx.HTTPTransport(retries=3),
        )

    def _call_responses_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call the Codex Responses API with messages."""
        # Convert messages to Responses API input format
        input_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                input_messages.append({
                    "role": "developer",
                    "content": content if isinstance(content, str) else json.dumps(content)
                })
            elif role == "assistant":
                # Check if this is a tool call message
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # Convert tool calls to Responses API format
                    for tc in tool_calls:
                        input_messages.append({
                            "type": "function_call",
                            "id": tc.get("id", f"call_{int(time.time())}"),
                            "call_id": tc.get("id", f"call_{int(time.time())}"),
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        })
                else:
                    input_messages.append({
                        "role": "assistant",
                        "content": content if isinstance(content, str) else json.dumps(content)
                    })
            elif role == "tool":
                # Tool result message
                input_messages.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else json.dumps(content),
                })
            else:
                input_messages.append({
                    "role": "user",
                    "content": content if isinstance(content, str) else json.dumps(content)
                })

        payload = {
            "model": self.model,
            "input": input_messages,
        }

        # Add tools if provided
        if tools:
            responses_tools = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    responses_tools.append({
                        "type": "function",
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    })
            if responses_tools:
                payload["tools"] = responses_tools

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        url = f"{self.base_url}/responses"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.http_client.post(
                    url, json=payload, headers=headers
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait = min(2 ** attempt * 5, 30)
                    time.sleep(wait)
                    continue
                elif response.status_code >= 500:
                    wait = min(2 ** attempt * 3, 15)
                    time.sleep(wait)
                    continue
                else:
                    raise Exception(
                        f"Codex API error {response.status_code}: {response.text[:500]}"
                    )
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise
            except httpx.ConnectError:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise

        raise Exception("Max retries exceeded for Codex API")

    def _responses_to_chat_completion(
        self, resp: Dict[str, Any]
    ) -> ChatCompletion:
        """Convert Responses API response to ChatCompletion format."""
        # Extract text output
        output_text = ""
        tool_calls_list = []

        output = resp.get("output", [])
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "message":
                        # Text message
                        content = item.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    output_text += c.get("text", "")
                        elif isinstance(content, str):
                            output_text += content
                    elif item_type == "function_call":
                        # Tool call
                        from openai.types.chat.chat_completion_message_tool_call import (
                            ChatCompletionMessageToolCall,
                            Function,
                        )
                        tool_calls_list.append(
                            ChatCompletionMessageToolCall(
                                id=item.get("call_id", item.get("id", f"call_{int(time.time())}")),
                                type="function",
                                function=Function(
                                    name=item.get("name", ""),
                                    arguments=item.get("arguments", "{}"),
                                ),
                            )
                        )
        elif isinstance(output, str):
            output_text = output

        # Also check output_text field directly
        if not output_text and not tool_calls_list:
            output_text = resp.get("output_text", "") or ""

        # Build ChatCompletion
        message = ChatCompletionMessage(
            role="assistant",
            content=output_text if output_text else None,
            tool_calls=tool_calls_list if tool_calls_list else None,
        )

        usage_data = resp.get("usage", {})
        usage = CompletionUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens",
                usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)),
        )

        return ChatCompletion(
            id=resp.get("id", f"chatcmpl-{int(time.time())}"),
            choices=[
                Choice(
                    finish_reason="tool_calls" if tool_calls_list else "stop",
                    index=0,
                    message=message,
                )
            ],
            created=int(time.time()),
            model=self.model,
            object="chat.completion",
            usage=usage,
        )


class FakeChatCompletions:
    """Mimics openai.chat.completions interface."""

    def __init__(self, client: CodexResponsesClient):
        self._client = client

    def create(self, messages, model=None, tools=None, **kwargs) -> ChatCompletion:
        # Convert OpenAI message format to dict
        msg_list = []
        for m in messages:
            if isinstance(m, dict):
                msg_list.append(m)
            else:
                msg_list.append(dict(m))

        resp = self._client._call_responses_api(msg_list, tools=tools)
        return self._client._responses_to_chat_completion(resp)


class FakeChat:
    """Mimics openai.chat interface."""

    def __init__(self, client: CodexResponsesClient):
        self.completions = FakeChatCompletions(client)


class FakeOpenAIClient:
    """Mimics openai.OpenAI client interface for CAMEL compatibility."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._codex = CodexResponsesClient(base_url, api_key, model)
        self.chat = FakeChat(self._codex)


# ============================================================
# CAMEL Model Backend using Codex Responses API
# ============================================================

class CodexModelBackend(BaseModelBackend):
    """A CAMEL model backend that uses the Codex Responses API."""

    def __init__(
        self,
        model_type: str = "gpt-5.1-codex-mini",
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        model_config_dict: Optional[Dict[str, Any]] = None,
    ):
        _api_key = api_key or os.environ.get("CODEX_API_KEY", "")
        _url = url or os.environ.get("CODEX_BASE_URL", "")
        _config = model_config_dict or {"temperature": 0}

        # Call parent init
        super().__init__(
            model_type=model_type,
            model_config_dict=_config,
            api_key=_api_key,
            url=_url,
        )

        # Create the fake client
        self._client = FakeOpenAIClient(
            base_url=_url,
            api_key=_api_key,
            model=model_type,
        )

        # Token tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def token_counter(self):
        return None

    @property
    def stream(self) -> bool:
        return False

    def check_model_config(self):
        pass

    def _run(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Run the model with the given messages."""
        result = self._client.chat.completions.create(
            messages=messages,
            model=str(self.model_type),
            tools=tools,
        )

        # Track tokens
        if result.usage:
            self._total_input_tokens += result.usage.prompt_tokens
            self._total_output_tokens += result.usage.completion_tokens

        return result

    async def _arun(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Async version - just calls sync for now."""
        return self._run(messages, response_format, tools)

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens


def create_codex_model(
    model_type: str = "gpt-5.1-codex-mini",
    api_key: Optional[str] = None,
    url: Optional[str] = None,
    temperature: float = 0,
) -> CodexModelBackend:
    """Factory function to create a Codex model backend."""
    return CodexModelBackend(
        model_type=model_type,
        api_key=api_key,
        url=url,
        model_config_dict={"temperature": temperature},
    )
