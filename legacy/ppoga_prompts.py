"""
PPoGA Prompts Module

PPoGA 시스템에서 사용되는 모든 LLM 프롬프트를 정의하는 모듈입니다.
"""

# 1. 계획 수립 프롬프트
DECOMPOSE_PLAN_PROMPT = """당신은 복잡한 질문을 해결하기 위한 전략가입니다. 주어진 질문을 해결하기 위한 단계별 실행 계획을 수립해야 합니다.

질문: {question}

계획을 JSON 형식으로 작성하세요:
{{
    "plan": [
        {{"description": "첫 번째 단계"}},
        {{"description": "두 번째 단계"}}
    ],
    "rationale": "계획 수립의 근거"
}}"""

# 2. 예측 프롬프트
PREDICT_PROMPT = """다음 계획 단계를 실행했을 때 예상되는 결과를 예측하세요.

현재 계획 단계: {plan_step}

예측을 JSON 형식으로 작성하세요:
{{
    "success_scenario": "성공 시나리오",
    "failure_scenario": "실패 시나리오",
    "confidence": "high|medium|low"
}}"""

# 3. 최종 답변 생성 프롬프트
FINAL_ANSWER_PROMPT = """수집된 정보를 바탕으로 최종 답변을 생성하세요.

질문: {question}
수집된 정보: {knowledge_summary}

답변을 JSON 형식으로 작성하세요:
{{
    "answer": "최종 답변",
    "confidence": "high|medium|low",
    "reasoning": "추론 과정"
}}"""
