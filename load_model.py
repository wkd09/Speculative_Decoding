from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, TypedDict

SMALL_MODEL_ID: Final = "mlx-community/Qwen3-0.6B-4bit"
BIG_MODEL_ID: Final = "mlx-community/Qwen3.5-4B-OptiQ-4bit"


class ChatMessage(TypedDict):
    role: str
    content: str


class MlxModel(Protocol):
    pass


class MlxTokenizer(Protocol):
    chat_template: str | None

    def apply_chat_template(
        self,
        messages: list[ChatMessage],
        *,
        add_generation_prompt: bool,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class LoadedMlxLM:
    tokenizer: MlxTokenizer
    model: MlxModel
    model_id: str


def load_small() -> LoadedMlxLM:
    return load_mlx_lm(SMALL_MODEL_ID)


def load_big() -> LoadedMlxLM:
    return load_mlx_lm(BIG_MODEL_ID)


def load_mlx_lm(model_id: str) -> LoadedMlxLM:
    from mlx_lm import load

    model, tokenizer = load(model_id)
    return LoadedMlxLM(tokenizer=tokenizer, model=model, model_id=model_id)
