import json
import os
from collections.abc import Iterator
from typing import Any

os.environ["LITELLM_OFFLINE_MODE"] = "true"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"


import httpx

# pyrefly: ignore[missing-import]
import litellm as sdk

# pyrefly: ignore[missing-import]
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

# pyrefly: ignore[missing-import]
from litellm.types.utils import ChatCompletionDeltaToolCall

from enboite.lib.llm import types


def completion(
    model: str,
    messages: list[dict[Any, Any]],
    reasoning_effort: str|None,
    tools: list[dict[str, Any]]|None = None,
    api_base: str|None = None,
    api_key: str|None = None,
    proxy: str | None = None,
    seed: int|None = None
) -> Iterator[types.StreamEvent]:
    
    if proxy:
        sdk.client_session = httpx.Client(
            proxy=proxy
        )
        sdk.aclient_session = httpx.AsyncClient(
            proxy=proxy
        )
    
    kwargs: Any = {}
    if not reasoning_effort is None:
        kwargs["reasoning_effort"] = reasoning_effort
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if tools:
        kwargs["tools"] = tools
    
    if not seed is None:
        kwargs["seed"] = seed
    
    # pyrefly: ignore[bad-assignment]
    _: CustomStreamWrapper = sdk.completion(
        model=model,
        messages=messages,
        stream=True,
        **kwargs
    )
    for chunk in _:
        if not chunk.choices:
            continue
        
        choice = chunk.choices[0]
        delta = choice.delta
        
        reasoning = delta.reasoning_content if hasattr(delta, "reasoning_content") else None
        content = delta.content if hasattr(delta, "content") else None
        tool_calls = delta.tool_calls if hasattr(delta, "tool_calls") else None
        
        if reasoning:
            yield types.StreamReasoning(reasoning)
        if content:
            yield types.StreamContent(content)
        
        if tool_calls:
            for tool in tool_calls:
                if isinstance(tool, ChatCompletionDeltaToolCall):
                    yield types.StreamToolCall(
                        index=tool.index,
                        id=tool.id,
                        function=str(tool.function.name),
                        arguments=json.loads(tool.function.arguments),
                    )
                else:
                    raise TypeError(f"unknown tool call function type: {tool}")
        
        if choice.finish_reason:
            yield types.StreamDone(str(choice.finish_reason))