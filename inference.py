from __future__ import annotations

from dataclasses import dataclass

from speculative_decoding.load_model import ChatMessage, MlxModel, MlxTokenizer


@dataclass(frozen=True, slots=True)
class SpeculativeResult:
    # 최종 speculative decoding 결과 텍스트
    text: str
    # 실제 선택된 토큰들의 평균 log-likelihood
    average_log_likelihood: float
    # draft model 제안이 그대로 채택된 토큰 수
    accepted_token_count: int
    # draft model 제안이 거절되고 target model 결과로 대체된 토큰 수
    rejected_token_count: int

    @property
    def acceptance_rate(self) -> float:
        # acceptance rate = accept / (accept + reject)
        total_token_count = self.accepted_token_count + self.rejected_token_count
        if total_token_count == 0:
            return 0.0
        return self.accepted_token_count / total_token_count


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
    # 실제 speculative decoding은 big tokenizer 기준으로 진행하고,
    # small model은 draft model 역할만 맡긴다.
    _ = small_tokenizer
    return generate_speculative_text(
        draft_model=small_model,
        model=big_model,
        tokenizer=big_tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )


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


def generate_speculative_text(
    draft_model: MlxModel,
    model: MlxModel,
    tokenizer: MlxTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> SpeculativeResult:
    from mlx_lm import stream_generate

    # stream_generate는 토큰이 생성될 때마다 잘린 텍스트 조각을 주므로
    # 마지막에 join해서 전체 문장을 복원한다.
    output_segments: list[str] = []
    # 각 step에서 최종 선택된 토큰의 log-likelihood를 모아 평균을 낸다.
    selected_log_likelihoods: list[float] = []
    accepted_token_count = 0
    rejected_token_count = 0

    for response in stream_generate(
        model,
        tokenizer,
        prompt=_chat_prompt(tokenizer, prompt),
        draft_model=draft_model,
        max_tokens=max_new_tokens,
    ):
        # 응답 텍스트는 누적 조각이 아니라 "이번 step에서 새로 확정된 부분"이다.
        output_segments.append(response.text)
        # response.token은 최종 선택된 토큰 id이고,
        # response.logprobs[token]은 그 토큰의 log-probability다.
        selected_log_likelihoods.append(float(response.logprobs[response.token].item()))
        # from_draft=True면 small model의 draft 토큰이 accept된 것이다.
        if response.from_draft:
            accepted_token_count += 1
        # from_draft=False면 draft가 reject되고 big model 결과가 사용된 것이다.
        else:
            rejected_token_count += 1

    average_log_likelihood = 0.0
    if selected_log_likelihoods:
        # verification score처럼 가볍게 보기 위한 평균값
        average_log_likelihood = sum(selected_log_likelihoods) / len(selected_log_likelihoods)

    return SpeculativeResult(
        text="".join(output_segments),
        average_log_likelihood=average_log_likelihood,
        accepted_token_count=accepted_token_count,
        rejected_token_count=rejected_token_count,
    )


def _chat_prompt(tokenizer: MlxTokenizer, prompt: str) -> str:
    if tokenizer.chat_template is None:
        return prompt
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
