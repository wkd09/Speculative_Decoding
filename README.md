# Speculative Decoding with MLX

Apple Silicon에서 MLX-LM 모델로 일반 생성과 draft-model 기반 생성을 비교하는 작은 실험 코드입니다.

## Models

| Role | Model |
|---|---|
| draft | `mlx-community/Qwen3-0.6B-4bit` |
| target | `mlx-community/Qwen3.5-4B-OptiQ-4bit` |

## Setup

```bash
.venv/bin/uv pip install -r speculative_decoding/requirements.txt
```

## Run

```bash
.venv/bin/python speculative_decoding/test.py
```

The first run may download the MLX model weights from Hugging Face.

## Current Result

Recent local run:

```text
Loaded draft MLX model: mlx-community/Qwen3-0.6B-4bit
Loaded target MLX model: mlx-community/Qwen3.5-4B-OptiQ-4bit
[1] normal latency: 9.5089s
[1] speculative latency: 1.0100s
[1] verification score: 0.0000
[2] normal latency: 7.5915s
[2] speculative latency: 0.9762s
[2] verification score: 0.0000
[3] normal latency: 7.4950s
[3] speculative latency: 1.0651s
[3] verification score: 0.0000
[4] normal latency: 7.6939s
[4] speculative latency: 0.9861s
[4] verification score: 0.0000
[5] normal latency: 7.5037s
[5] speculative latency: 0.9864s
[5] verification score: 0.0000
Average Normal Inference Latency: 7.9586 seconds
Average Speculative Decoding Latency: 1.0048 seconds
Average Normal Inference Tokens per second: 25.34 tokens/second
Average Speculative Decoding Tokens per second: 199.25 tokens/second
```

## Notes

This is not full token-level speculative decoding yet. The current `speculative_decoding()` path generates draft text with the small model and reports a placeholder verification score of `0.0`. The latency comparison is still useful for checking MLX model loading and generation speed, but acceptance/rejection against the big model needs to be implemented separately.

## Files

- `load_model.py`: MLX model IDs and loaders
- `inference.py`: MLX text generation helpers
- `latency.py`: latency and tokens/sec measurement
- `test.py`: executable experiment script
