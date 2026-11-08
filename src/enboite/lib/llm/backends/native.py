import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import requests

from enboite.lib.llm import types

try:
    # pyrefly: ignore[missing-import]
    import orjson as _json
except ImportError:
    import json as _json


def completion(
    model: str,
    messages: list[dict[Any, Any]],
    reasoning_effort: str|None,
    tools: list[dict[str, Any]] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    proxy: str|None = None,
    seed: int|None = None
) -> Iterator[types.StreamEvent]:
    
    _messages = deepcopy(messages)
    
    for message in _messages:
        for tool_call in message.get("tool_calls", []):
            if type(tool_call["function"]["arguments"]) is str:
                arguments = tool_call["function"]["arguments"]
                tool_call["function"]["arguments"] = json.loads(arguments)
    
    http = requests.Session()
    http.headers.update({
        "User-Agent": "EnBoite/0.0.0"
    })
    if proxy:
        http.proxies.update(
            {
                "http": proxy,
                "https": proxy
            }
        )
    
    kwargs: dict[str, Any] = {}
    if reasoning_effort:
        if reasoning_effort == "none":
            kwargs["think"] = False
        else:
            kwargs["think"] = reasoning_effort
    
    if tools:
        kwargs["tools"] = tools
    
    if not seed is None:
        kwargs["options"] = {"seed": seed}
    
    Stream = http.post(
        f"{api_base}/api/chat",
        json={
            "model": model,
            "messages": _messages,
            "stream": True,
            **kwargs
        },
        stream=True
    )
    
    if Stream.status_code != 200:
        print(Stream.text)
        Stream.raise_for_status()
    
    for line in Stream.iter_lines(chunk_size=1):
        line: bytes
        if not line:
            continue
        
        data = _json.loads(line)
        message = data.get("message", {})
        
        if chunk := message.get("thinking"):
            yield types.StreamReasoning(chunk)
        
        if chunk := message.get("content"):
            yield types.StreamContent(chunk)
        
        if calls := message.get("tool_calls"):
            for i in calls:
                yield types.StreamToolCall(
                    index=i["function"]["index"],
                    id=i["id"],
                    function=i["function"]["name"],
                    arguments=i["function"]["arguments"],
                )
        
        if data.get("done"):
            yield types.StreamDone(
                data["done_reason"],
                total_tokens=(
                    data["prompt_eval_count"] + data["eval_count"]
                )
            )