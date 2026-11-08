import json
from collections.abc import Iterator
from typing import Any

from openai import DefaultHttpx2Client, OpenAI, Stream
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from enboite.lib.llm import types


def completion(
    model: str,
    messages: list[ChatCompletionMessageParam],
    reasoning_effort: str|None,
    tools: list[dict[str, Any]] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    proxy: str | None = None,
    seed: int | None = None
) -> Iterator[types.StreamEvent]:
    
    kwargs = {}
    if proxy:
        kwargs["http_client"] = DefaultHttpx2Client(
            proxy=proxy
        )
    
    client = OpenAI(
        base_url=api_base,
        api_key=api_key or "EMPTY",
        max_retries=0,
        **kwargs
    )
    
    kwargs: dict[str, Any] = {}
    
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    
    if tools:
        kwargs["tools"] = tools
    
    if not seed is None:
        kwargs["seed"] = seed
    
    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={
            "include_usage": True,
        },
        **kwargs,
    )
    total_tokens = 0
    finish_reason = None
    for chunk in stream:
        if chunk.usage is not None:
            total_tokens = chunk.usage.total_tokens
        
        if not chunk.choices:
            continue
        
        choice = chunk.choices[0]
        delta = choice.delta
        
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning is None:
            reasoning = getattr(delta, "reasoning", None)
        
        content = delta.content
        
        tool_calls = delta.tool_calls
        
        if reasoning:
            yield types.StreamReasoning(reasoning)
        
        if content:
            yield types.StreamContent(content)
        
        # if tool_calls:
        #     print(tool_calls)
        
        if tool_calls:
            for tool in tool_calls:
                yield types.StreamToolCall(
                    index=tool.index,
                    id=tool.id,
                    # pyrefly: ignore[missing-attribute]
                    function=tool.function.name,
                    # pyrefly: ignore[missing-attribute]
                    arguments=json.loads(tool.function.arguments),
                )
        
        if choice.finish_reason:
            finish_reason = str(choice.finish_reason)
    
    yield types.StreamDone(
        str(finish_reason),
        total_tokens=total_tokens
    )