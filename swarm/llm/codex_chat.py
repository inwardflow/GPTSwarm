#!/usr/bin/env python3
"""
Codex Responses API adapter for GPTSwarm.
Wraps the OpenAI Responses API (used by gpt-5.x-codex models) to be compatible
with GPTSwarm's LLM interface which expects chat/completions style interaction.
"""

import asyncio
import os
from dataclasses import asdict
from typing import List, Union, Optional, Dict, Any
from dotenv import load_dotenv
import random

from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry
from swarm.utils.globals import Cost, PromptTokens, CompletionTokens

load_dotenv()

CODEX_API_KEY = os.getenv("OPENAI_API_KEY", "")
CODEX_BASE_URL = os.getenv("OPENAI_BASE_URL", "")


def _extract_text_from_response(response) -> str:
    """Extract text content from a Responses API response object."""
    for item in response.output:
        if hasattr(item, 'content') and item.content:
            for content_piece in item.content:
                if hasattr(content_piece, 'text'):
                    return content_piece.text
    return ""


def _count_tokens(response):
    """Extract token usage from response."""
    if hasattr(response, 'usage') and response.usage:
        prompt_tokens = getattr(response.usage, 'input_tokens', 0)
        completion_tokens = getattr(response.usage, 'output_tokens', 0)
        return prompt_tokens, completion_tokens
    return 0, 0


def codex_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps: int = 1,
    return_cost: bool = False,
) -> Union[List[str], str]:
    """Synchronous call to the Codex Responses API."""
    if messages[0].content == '$skip$':
        return ''

    client = OpenAI(api_key=CODEX_API_KEY, base_url=CODEX_BASE_URL)

    # Convert Message objects to dict format for the Responses API
    input_messages = [asdict(message) for message in messages]

    response = client.responses.create(
        model=model,
        input=input_messages,
    )

    text = _extract_text_from_response(response)
    prompt_tokens, completion_tokens = _count_tokens(response)

    # Update global counters
    PromptTokens.instance().value += prompt_tokens
    CompletionTokens.instance().value += completion_tokens

    if num_comps == 1:
        return text
    return [text]


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(5))
async def codex_achat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps: int = 1,
    return_cost: bool = False,
) -> Union[List[str], str]:
    """Async call to the Codex Responses API."""
    if messages[0].content == '$skip$':
        return ''

    aclient = AsyncOpenAI(api_key=CODEX_API_KEY, base_url=CODEX_BASE_URL)

    # Convert Message objects to dict format for the Responses API
    input_messages = [asdict(message) for message in messages]

    response = await aclient.responses.create(
        model=model,
        input=input_messages,
    )

    text = _extract_text_from_response(response)
    prompt_tokens, completion_tokens = _count_tokens(response)

    # Update global counters
    PromptTokens.instance().value += prompt_tokens
    CompletionTokens.instance().value += completion_tokens

    if num_comps == 1:
        return text
    return [text]


@LLMRegistry.register('CodexChat')
class CodexChat(LLM):
    """LLM implementation for Codex Responses API models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return await codex_achat(
            self.model_name,
            messages,
            max_tokens,
            temperature,
            num_comps
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return codex_chat(
            self.model_name,
            messages,
            max_tokens,
            temperature,
            num_comps
        )
