from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StreamContent:
    content: str


@dataclass(frozen=True, slots=True)
class StreamReasoning:
    content: str


@dataclass(frozen=True, slots=True)
class StreamToolCall:
    index: int
    function: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True, slots=True)
class StreamDone:
    content: str
    total_tokens: int = 0

type StreamEvent = (
    StreamContent |
    StreamReasoning |
    StreamToolCall |
    StreamDone
)