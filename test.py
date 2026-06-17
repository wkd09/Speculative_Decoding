from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speculative_decoding.latency import LatencyResult, measure_latency
from speculative_decoding.load_model import BIG_MODEL_ID, SMALL_MODEL_ID, load_big, load_small

PROMPTS: Final = (
    "The future of artificial intelligence is ",
    "Machine learning is transforming the world by ",
    "Natural language processing enables computers to understand ",
    "Generative models like GPT-3 can create ",
    "AI ethics and fairness are important considerations for ",
)
MAX_NEW_TOKENS: Final = 200


def main() -> None:
    small = load_small()
    big = load_big()
    print(f"Loaded small MLX model: {SMALL_MODEL_ID}")
    print(f"Loaded big MLX model: {BIG_MODEL_ID}")
    results = [
        measure_latency(
            small.model,
            small.tokenizer,
            big.model,
            big.tokenizer,
            prompt,
            MAX_NEW_TOKENS,
        )
        for prompt in PROMPTS
    ]
    _print_each_result(results)
    _print_average_result(results)


def _print_each_result(results: list[LatencyResult]) -> None:
    for index, result in enumerate(results, start=1):
        print(f"[{index}] normal latency: {result.normal_latency_seconds:.4f}s")
        print(f"[{index}] speculative latency: {result.speculative_latency_seconds:.4f}s")
        print(f"[{index}] verification score: {result.verification_score:.4f}")


def _print_average_result(results: list[LatencyResult]) -> None:
    result_count = len(results)
    average_normal_latency = sum(item.normal_latency_seconds for item in results) / result_count
    average_speculative_latency = sum(item.speculative_latency_seconds for item in results) / result_count
    average_normal_tps = sum(item.normal_tokens_per_second for item in results) / result_count
    average_speculative_tps = sum(item.speculative_tokens_per_second for item in results) / result_count
    print(f"Average Normal Inference Latency: {average_normal_latency:.4f} seconds")
    print(f"Average Speculative Decoding Latency: {average_speculative_latency:.4f} seconds")
    print(f"Average Normal Inference Tokens per second: {average_normal_tps:.2f} tokens/second")
    print(f"Average Speculative Decoding Tokens per second: {average_speculative_tps:.2f} tokens/second")


if __name__ == "__main__":
    main()
