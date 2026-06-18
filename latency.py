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
    accepted_token_count: int
    rejected_token_count: int
    acceptance_rate: float


def measure_latency(
    small_model: MlxModel,
    small_tokenizer: MlxTokenizer,
    big_model: MlxModel,
    big_tokenizer: MlxTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> LatencyResult:
    # target model만 단독으로 돌렸을 때의 기준 latency
    normal_start = perf_counter()
    normal_text = normal_inference(big_model, big_tokenizer, prompt, max_new_tokens)
    normal_latency_seconds = perf_counter() - normal_start

    # draft + target 조합의 speculative decoding latency
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
        speculative_text=speculative_result.text,
        normal_latency_seconds=normal_latency_seconds,
        speculative_latency_seconds=speculative_latency_seconds,
        normal_tokens_per_second=_tokens_per_second(max_new_tokens, normal_latency_seconds),
        speculative_tokens_per_second=_tokens_per_second(max_new_tokens, speculative_latency_seconds),
        verification_score=speculative_result.average_log_likelihood,
        accepted_token_count=speculative_result.accepted_token_count,
        rejected_token_count=speculative_result.rejected_token_count,
        acceptance_rate=speculative_result.acceptance_rate,
    )


def _tokens_per_second(token_count: int, elapsed_seconds: float) -> float:
    # 0으로 나누는 경우를 방지
    if elapsed_seconds <= 0:
        return 0.0
    return token_count / elapsed_seconds
