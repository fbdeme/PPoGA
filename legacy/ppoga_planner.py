"""
PPoGA Planner Module

PPoGA의 전략가(Planner) 역할을 담당하는 모듈입니다.
계획 수립, 예측, 사고, 평가, 자기 교정을 수행합니다.
"""

import json
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# PoG 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PoG'))

from ppoga_memory import PPoGAMemory, PlanStep
from ppoga_prompts import DECOMPOSE_PLAN_PROMPT, PREDICT_PROMPT, FINAL_ANSWER_PROMPT

try:
    from utils import run_llm, extract_reason_and_anwer
    from prompt_list import extract_relation_prompt
except ImportError:
    print("Warning: Could not import PoG modules. Some functions may not work.")


class PPoGAPlanner:
    """
    PPoGA의 전략가(Planner) 클래스
    
    주요 역할:
    1. 계획 수립 (Planning)
    2. 예측 (Prediction) 
    3. 사고 및 평가 (Thinking & Evaluation)
    4. 자기 교정 (Self-Correction)
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Planner 초기화
        
        Args:
            llm_config: LLM 설정 (model, temperature, api_key 등)
        """
        self.llm_config = llm_config
        self.model = llm_config.get("model", "gpt-3.5-turbo")
        self.temperature = llm_config.get("temperature", 0.3)
        self.max_tokens = llm_config.get("max_tokens", 2048)
        self.api_key = llm_config.get("api_key", "")
    
    def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> Tuple[str, Dict[str, int]]:
        """LLM을 호출하는 내부 메서드"""
        temp = temperature if temperature is not None else self.temperature
        
        try:
            return run_llm(
                prompt=prompt,
                temperature=temp,
                max_tokens=self.max_tokens,
                opeani_api_keys=self.api_key,
                engine=self.model,
                print_in=False,
                print_out=False
            )
        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return f"Error: {str(e)}", {"total": 0, "input": 0, "output": 0}
    
    def plan(self, question: str, memory: PPoGAMemory) -> bool:
        """
        1. 계획 수립 (Planning)
        
        Args:
            question: 사용자 질문
            memory: PPoGA 메모리 객체
            
        Returns:
            bool: 계획 수립 성공 여부
        """
        print(f"🧐 Planner: 질문 분석 및 계획 수립 중...")
        
        prompt = DECOMPOSE_PLAN_PROMPT.format(question=question)
        response, token_info = self._call_llm(prompt)
        memory.increment_llm_calls()
        
        try:
            # JSON 응답 파싱
            plan_data = json.loads(response)
            plan_steps = plan_data.get("plan", [])
            rationale = plan_data.get("rationale", "")
            
            if not plan_steps:
                print("❌ 계획 수립 실패: 빈 계획")
                return False
            
            # 메모리에 계획 저장
            memory.add_plan(plan_steps, rationale)
            
            print(f"✅ 계획 수립 완료: {len(plan_steps)}개 단계")
            for i, step in enumerate(plan_steps):
                print(f"   {i+1}. {step['description']}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ 계획 파싱 오류: {e}")
            print(f"응답: {response}")
            return False
    
    def predict(self, current_step: PlanStep, memory: PPoGAMemory) -> str:
        """
        2. 예측 (Prediction)
        
        Args:
            current_step: 현재 실행할 계획 단계
            memory: PPoGA 메모리 객체
            
        Returns:
            str: 예측 결과
        """
        print(f"🔮 Planner: 단계 {current_step.id} 결과 예측 중...")
        
        prompt = PREDICT_PROMPT.format(plan_step=current_step.description)
        response, token_info = self._call_llm(prompt)
        memory.increment_llm_calls()
        
        try:
            prediction_data = json.loads(response)
            prediction_text = f"성공: {prediction_data.get('success_scenario', '')}, " \
                            f"실패: {prediction_data.get('failure_scenario', '')}, " \
                            f"신뢰도: {prediction_data.get('confidence', 'medium')}"
            
            print(f"✅ 예측 완료: {prediction_text}")
            return prediction_text
            
        except json.JSONDecodeError:
            print(f"⚠️ 예측 파싱 실패, 원본 응답 사용")
            return response
    
    def select_relations(self, entity_name: str, entity_id: str, 
                        available_relations: List[str], current_step: PlanStep) -> List[str]:
        """
        3A. 탐색 경로 결정 (관계 선택)
        
        Args:
            entity_name: 엔티티 이름
            entity_id: 엔티티 ID
            available_relations: 사용 가능한 관계 목록
            current_step: 현재 계획 단계
            
        Returns:
            List[str]: 선택된 관계 목록
        """
        print(f"🎯 Planner: {entity_name}에 대한 관계 선택 중...")
        
        # PoG의 extract_relation_prompt 재사용
        relations_str = "; ".join(available_relations)
        prompt = f"""현재 계획 단계: {current_step.description}
엔티티: {entity_name}
사용 가능한 관계들: {relations_str}

이 계획 단계를 해결하는 데 가장 관련성이 높은 관계들을 최대 3개까지 선택하세요.
선택된 관계들을 리스트 형태로 반환하세요: ["relation1", "relation2"]"""
        
        response, token_info = self._call_llm(prompt)
        
        try:
            # 간단한 파싱 (리스트 형태 추출)
            if "[" in response and "]" in response:
                start = response.find("[")
                end = response.rfind("]") + 1
                relations_list = json.loads(response[start:end])
                print(f"✅ 관계 선택 완료: {relations_list}")
                return relations_list
            else:
                # 첫 번째 관계만 선택
                selected = [available_relations[0]] if available_relations else []
                print(f"⚠️ 파싱 실패, 첫 번째 관계 선택: {selected}")
                return selected
                
        except Exception as e:
            print(f"⚠️ 관계 선택 오류: {e}, 첫 번째 관계 사용")
            return [available_relations[0]] if available_relations else []
    
    def think_and_evaluate(self, prediction: str, observation: str, 
                          current_step: PlanStep, memory: PPoGAMemory) -> str:
        """
        4. 사고 및 평가 (Thinking & Evaluation)
        
        Args:
            prediction: 이전 예측
            observation: 실제 관찰 결과
            current_step: 현재 계획 단계
            memory: PPoGA 메모리 객체
            
        Returns:
            str: 다음 행동 ('PROCEED', 'CORRECT_PATH', 'REPLAN', 'FINISH')
        """
        print(f"🤔 Planner: 결과 분석 및 다음 행동 결정 중...")
        
        memory_summary = memory.get_memory_summary()
        
        prompt = f"""예측: {prediction}
실제 관찰: {observation}
현재 단계: {current_step.description}
메모리 요약: {memory_summary}

예측과 실제 결과를 비교하고 다음 행동을 결정하세요:
- PROCEED: 다음 계획 단계로 진행
- CORRECT_PATH: 현재 단계를 다른 방법으로 재시도  
- REPLAN: 전체 계획을 재수립
- FINISH: 충분한 정보를 얻어 답변 가능

다음 행동을 JSON 형식으로 반환하세요:
{{"next_action": "PROCEED|CORRECT_PATH|REPLAN|FINISH", "reasoning": "이유"}}"""
        
        response, token_info = self._call_llm(prompt)
        memory.increment_llm_calls()
        
        try:
            result = json.loads(response)
            next_action = result.get("next_action", "PROCEED")
            reasoning = result.get("reasoning", "")
            
            print(f"✅ 다음 행동 결정: {next_action}")
            print(f"   이유: {reasoning}")
            
            # 사고 내용을 메모리에 기록
            thought = f"예측 vs 실제: {prediction[:50]}... vs {observation[:50]}... | 결정: {next_action} | 이유: {reasoning}"
            memory.update_cycle_thought(current_step.id, thought)
            
            return next_action
            
        except json.JSONDecodeError:
            print(f"⚠️ 평가 파싱 실패, 기본값 PROCEED 사용")
            return "PROCEED"
    
    def replan(self, question: str, memory: PPoGAMemory, failure_reason: str) -> bool:
        """
        5. 전략적 계획 수정 (Strategic Replanning)
        
        Args:
            question: 원래 질문
            memory: PPoGA 메모리 객체
            failure_reason: 실패 이유
            
        Returns:
            bool: 재계획 성공 여부
        """
        print(f"🔄 Planner: 전략적 재계획 수행 중...")
        
        memory_summary = memory.get_memory_summary()
        failed_plan = [step.description for step in memory.strategy["overall_plan"]]
        
        prompt = f"""원래 질문: {question}
실패한 계획: {failed_plan}
실패 이유: {failure_reason}
현재까지 수집된 정보: {memory_summary}

이전 실패를 바탕으로 더 효과적인 새로운 계획을 수립하세요.

새로운 계획을 JSON 형식으로 작성하세요:
{{
    "plan": [
        {{"description": "새로운 첫 번째 단계"}},
        {{"description": "새로운 두 번째 단계"}}
    ],
    "rationale": "재계획의 근거와 이전 계획 대비 개선점"
}}"""
        
        response, token_info = self._call_llm(prompt)
        memory.increment_llm_calls()
        
        try:
            plan_data = json.loads(response)
            plan_steps = plan_data.get("plan", [])
            rationale = plan_data.get("rationale", "")
            
            if not plan_steps:
                print("❌ 재계획 실패: 빈 계획")
                return False
            
            # 메모리에 새로운 계획 저장 (기존 계획은 alternative_plans로 이동)
            memory.add_plan(plan_steps, rationale)
            
            print(f"✅ 재계획 완료: {len(plan_steps)}개 단계")
            for i, step in enumerate(plan_steps):
                print(f"   {i+1}. {step['description']}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ 재계획 파싱 오류: {e}")
            return False
    
    def generate_final_answer(self, question: str, memory: PPoGAMemory) -> Dict[str, Any]:
        """
        6. 최종 답변 생성
        
        Args:
            question: 원래 질문
            memory: PPoGA 메모리 객체
            
        Returns:
            Dict[str, Any]: 최종 답변 정보
        """
        print(f"📝 Planner: 최종 답변 생성 중...")
        
        knowledge_summary = {
            "entities_discovered": len(memory.knowledge["exploration_graph"]["nodes"]),
            "reasoning_paths": memory.knowledge["reasoning_paths"],
            "key_findings": list(memory.knowledge["exploration_graph"]["nodes"].keys())[:10]
        }
        
        prompt = FINAL_ANSWER_PROMPT.format(
            question=question,
            knowledge_summary=str(knowledge_summary)
        )
        
        response, token_info = self._call_llm(prompt)
        memory.increment_llm_calls()
        
        try:
            answer_data = json.loads(response)
            
            final_answer = {
                "answer": answer_data.get("answer", "답변을 생성할 수 없습니다."),
                "confidence": answer_data.get("confidence", "low"),
                "reasoning": answer_data.get("reasoning", ""),
                "memory_summary": memory.get_memory_summary()
            }
            
            print(f"✅ 최종 답변 생성 완료")
            print(f"   답변: {final_answer['answer']}")
            print(f"   신뢰도: {final_answer['confidence']}")
            
            return final_answer
            
        except json.JSONDecodeError:
            print(f"⚠️ 답변 파싱 실패, 원본 응답 사용")
            return {
                "answer": response,
                "confidence": "low",
                "reasoning": "파싱 실패로 인한 원본 응답",
                "memory_summary": memory.get_memory_summary()
            }
