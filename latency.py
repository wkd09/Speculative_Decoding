from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from speculative_decoding.inference import normal_inference, speculative_decoding
from speculative_decoding.load_model import MlxModel, MlxTokenizer


@dataclass(frozen=True, slots=True)
class LatencyResult:
    normal_text: str
    speculative_text: str
    normal_latency_seconds: float
    speculative_latency_seconds: float
    normal_tokens_per_second: float
    speculative_tokens_per_second: float
    verification_score: float


def measure_latency(
    small_model: MlxModel,
    small_tokenizer: MlxTokenizer,
    big_model: MlxModel,
    big_tokenizer: MlxTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> LatencyResult:
    normal_start = perf_counter()
    normal_text = normal_inference(big_model, big_tokenizer, prompt, max_new_tokens)
    normal_latency_seconds = perf_counter() - normal_start

    speculative_start = perf_counter()
    speculative_result = speculative_decoding(
        small_model,
        small_tokenizer,
        big_model,
        big_tokenizer,
        prompt,
        max_new_tokens,
    )
    speculative_latency_seconds = perf_counter() - speculative_start

    return LatencyResult(
        normal_text=normal_text,
        speculative_text=speculative_result.draft_text,
        normal_latency_seconds=normal_latency_seconds,
        speculative_latency_seconds=speculative_latency_seconds,
        normal_tokens_per_second=_tokens_per_second(max_new_tokens, normal_latency_seconds),
        speculative_tokens_per_second=_tokens_per_second(max_new_tokens, speculative_latency_seconds),
        verification_score=speculative_result.average_log_likelihood,
    )


def _tokens_per_second(token_count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        return 0.0
    return token_count / elapsed_seconds
