import json
import time
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, Literal

from enboite.lib.llm import types

try:
    # pyrefly: ignore [missing-import]
    import orjson
except ImportError as e:
    print(e)
    import json
    class orjson:
        @staticmethod
        def loads(x: Any):
            return json.loads(x)
        @staticmethod
        def dumps(x: Any):
            return json.dumps(x).encode()

from enboite.lib import tooling, tooling_external


class client:
    def __init__(
        self,
        model: str,
        reasoning_effort: None|str,
        api_base: str|None = None,
        api_key: str|None = None,
        system_prompt: str|None = None,
        ctx: int|None = None,
        tools: list[dict[Any, Any]]|None = None,
        proxy: str|None = None,
        timeout: None|int = None,
        printError: bool = True,
        backend: Literal[
            "litellm",
            "openai"
        ] = "openai",
        seed: int|None = None
    ) -> None:
        if backend == "litellm":
            from enboite.lib.llm.backends import litellm
            self.backend = litellm
        
        elif backend == "openai":
            from enboite.lib.llm.backends import openai
            self.backend = openai
        
        elif backend == "native":
            from enboite.lib.llm.backends import native
            self.backend = native
        
        else:
            raise RuntimeError("invalid backend")
        
        self.seed = seed
        self.ctx = ctx
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.reasoning_effort = reasoning_effort
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.messages: list[dict[Any, Any]] = []
        self.total_token = 0
        self.token_per_sec = 0.0
        self._systemPrompt()
        self.printError = printError
        self.estimation_tokenization_factor: float = 0
        self.last_time_add = 0.0
        self.spike_char = 0
    
    def add(self, role: str, content: str) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )
    
    def clear(self):
        self.messages = []
        self._systemPrompt()
    
    def export(self) -> bytes:
        messages = deepcopy(self.messages)
        
        if messages[0]["role"] == "system":
            del messages[0]
        else:
            raise RuntimeError("not found the system prompt:83")
        
        return orjson.dumps(messages)
    
    def load(self, x: bytes) -> None:
        self.messages = orjson.loads(x)
        self._systemPrompt()
    
    def _systemPrompt(self):
        if type(self.system_prompt) is str:
            data = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]
            data.extend(self.messages)
            self.messages = data
    
    def debug_chat(self):
        for i in self.messages:
            for key, value in i.items():
                print(f"{key} > {str(value)[:100]}")
    
    def _generate(self, content: str) -> Iterator[types.StreamEvent]:
        if content != "":
            self.messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        
        self.last_time_add = time.monotonic()
        self.spike_char = max(len(self.messages[-1]["content"]), self.spike_char)
        
        
        _ = self.backend.completion(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            # pyrefly: ignore[bad-argument-type]
            messages=self.messages.copy(),
            tools=self.tools,
            api_base=self.api_base,
            api_key=self.api_key,
            seed=self.seed
        )
        self.messages.append({"role": "assistant", "content": ""})
        for chunk in _:
            if isinstance(chunk, types.StreamContent):
                self.messages[-1]["content"] += chunk.content
            elif isinstance(chunk, types.StreamDone):
                self.total_token = chunk.total_tokens
            elif isinstance(chunk, types.StreamToolCall):
                if not type(self.messages[-1].get("tool_calls")) is list:
                    _tool_calls: list[dict[str, Any]] = []
                    self.messages[-1]["tool_calls"] = _tool_calls
                self.messages[-1]["tool_calls"].append(
                    {
                        "id": chunk.id,
                        "type": "function",
                        "function": {
                            "name": chunk.function,
                            "arguments": json.dumps(chunk.arguments)
                        }
                    }
                )
            
            yield chunk
    
    def generate(self, content: str) -> Iterator[types.StreamEvent]:
        while True:
            tool_calls: list[dict[str, Any]] = []
            stream = self._generate(content)
            content = ""
            
            for chunk in stream:
                yield chunk
                
                if isinstance(chunk, types.StreamToolCall):
                    tool_calls.append(
                        
                        {
                            "function": chunk.function,
                            "arguments": chunk.arguments,
                            "id": chunk.id
                        }
                        
                    )
            
            if len(tool_calls) == 0:
                break
            
            for call in tool_calls:
                ID = call["id"]
                name = call["function"]
                arguments = call["arguments"]
                
                start_time_tool = time.monotonic()
                try:
                    if getattr(tooling_external, name, None):
                        func = getattr(tooling_external, name)
                        tool_result = str(func(**arguments))
                    else:
                        func = getattr(tooling, name)
                        if func.__module__ != tooling.__name__:
                            raise ValueError("Tool externe")
                        tool_result = func(**arguments)
                except Exception as e:  # noqa: BLE001
                    tool_result = f"Tool '{name}' failed: {type(e).__name__}: {e}"
                    if self.printError:
                        print(f"\033[41m{tool_result}\033[0m")
                
                print(f"tool:time::{name} : {time.monotonic() - start_time_tool:.4f}")
                
                if type(tool_result) != str and type(tool_result) is dict and not tool_result.get("type") in ("images",):
                    print(f"\033[33m{name}: return a type : {type(tool_result)}\033[0m")
                
                images = None
                if type(tool_result) is dict and tool_result['type'] == "images":
                    images = tool_result["value"]
                    tool_result = tool_result.get('content', "")
                
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": ID,
                        "content": str(tool_result)
                    }
                )
                if images:
                    # ajouter la gestion du mime, car tous passe par image/png
                    image_parts = []
                    for img in images:
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img}"}
                            }
                        )
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Content of an image requested by a tool. This message was generated automatically; please ignore it."
                            },
                            *image_parts
                        ]
                    })
    
    def unload_llm(self) -> None:
        raise NotImplementedError()
        # pyrefly: ignore[unreachable]
        i = 0
        think_old = self.think
        keep_old = self.keep_alive
        self.think = False
        self.keep_alive = 0
        # pyrefly: ignore[missing-attribute]
        while self.http.get(f"{self.api_base}/api/ps").json()["models"]:
            i+= 0.5
            print("unload...")
            print(f"...({i}s)")
            # pyrefly: ignore[missing-attribute]
            self._get("Do not respond at all.")
            time.sleep(i)
        print("ok")
        self.think = think_old
        self.keep_alive = keep_old