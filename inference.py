from __future__ import annotations

from dataclasses import dataclass

from speculative_decoding.load_model import ChatMessage, MlxModel, MlxTokenizer


@dataclass(frozen=True, slots=True)
class SpeculativeResult:
    draft_text: str
    average_log_likelihood: float


def normal_inference(
    big_model: MlxModel,
    big_tokenizer: MlxTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    return generate_text(big_model, big_tokenizer, prompt, max_new_tokens)


def speculative_decoding(
    small_model: MlxModel,
    small_tokenizer: MlxTokenizer,
    big_model: MlxModel,
    big_tokenizer: MlxTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> SpeculativeResult:
    _ = big_model
    _ = big_tokenizer
    draft_text = generate_text(small_model, small_tokenizer, prompt, max_new_tokens)
    return SpeculativeResult(draft_text=draft_text, average_log_likelihood=0.0)


def generate_text(
    model: MlxModel,
    tokenizer: MlxTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    from mlx_lm import generate

    return generate(
        model,
        tokenizer,
        prompt=_chat_prompt(tokenizer, prompt),
        max_tokens=max_new_tokens,
        verbose=False,
    )


def _chat_prompt(tokenizer: MlxTokenizer, prompt: str) -> str:
    if tokenizer.chat_template is None:
        return prompt
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
