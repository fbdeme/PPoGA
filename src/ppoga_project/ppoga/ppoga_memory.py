"""
PPoGA Memory Module

통합 메모리 아키텍처를 구현하는 모듈입니다.
전략, 실행, 지식의 3계층으로 구성된 메모리 시스템을 제공합니다.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ExecutionCycle:
    """단일 실행 사이클 (예측-행동-관찰-사고)을 나타내는 데이터 클래스"""
    step_id: int
    prediction: str = ""
    action: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    thought: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanStep:
    """계획의 단일 단계를 나타내는 데이터 클래스"""
    id: int
    description: str
    status: str = "Pending"  # Pending, InProgress, Success, Failed
    attempts: int = 0


class PPoGAMemory:
    """
    PPoGA의 통합 메모리 시스템
    
    세 개의 논리적 계층으로 구성:
    1. 전략 계층 (Strategic Layer): 계획과 전략적 의사결정
    2. 실행 계층 (Execution Layer): 실행 과정의 상세 로그
    3. 지식 계층 (Knowledge Layer): 탐색된 지식과 사실들
    """
    
    def __init__(self, question: str):
        self.question = question
        self.created_at = time.time()
        
        # 1. 전략 계층 (Strategic Layer): "우리는 어디로 가는가?"
        self.strategy = {
            "initial_question": question,
            "overall_plan": [],  # List[PlanStep]
            "plan_rationale": "",
            "status": "Planning",  # Planning, Executing, Evaluating, Replanning, Finished
            "alternative_plans": [],  # 폐기되거나 수정된 이전 계획들
            "replan_count": 0
        }
        
        # 2. 실행 계층 (Execution Layer): "우리는 어떻게 가고 있는가?"
        self.execution = {
            "current_step_id": 0,
            "log": {},  # Key: step_id, Value: List[ExecutionCycle]
            "total_cycles": 0,
            "total_llm_calls": 0,
            "total_kg_queries": 0
        }
        
        # 3. 지식 계층 (Knowledge Layer): "우리는 무엇을 알고 있는가?"
        self.knowledge = {
            "topic_entities": {},  # 초기 주제 엔티티들
            "exploration_graph": {
                "nodes": {},  # entity_id -> entity_info
                "edges": {}   # (entity1, relation, entity2) -> edge_info
            },
            "reasoning_paths": [],  # 성공적인 추론 경로들
            "candidate_entities": {},  # step_id -> List[entity_id]
            "failed_paths": []  # 실패한 경로들 (학습용)
        }
    
    def add_plan(self, plan_steps: List[Dict[str, Any]], rationale: str) -> None:
        """전략 계층에 새로운 계획을 추가합니다."""
        # 기존 계획이 있다면 alternative_plans로 이동
        if self.strategy["overall_plan"]:
            self.strategy["alternative_plans"].append({
                "plan": self.strategy["overall_plan"].copy(),
                "rationale": self.strategy["plan_rationale"],
                "abandoned_at": time.time(),
                "reason": "Replanning"
            })
            self.strategy["replan_count"] += 1
        
        # 새로운 계획 설정
        self.strategy["overall_plan"] = [
            PlanStep(id=i, description=step["description"]) 
            for i, step in enumerate(plan_steps)
        ]
        self.strategy["plan_rationale"] = rationale
        self.strategy["status"] = "Executing"
        self.execution["current_step_id"] = 0
    
    def get_current_step(self) -> Optional[PlanStep]:
        """현재 실행 중인 계획 단계를 반환합니다."""
        if not self.strategy["overall_plan"]:
            return None
        
        current_id = self.execution["current_step_id"]
        if current_id >= len(self.strategy["overall_plan"]):
            return None
            
        return self.strategy["overall_plan"][current_id]
    
    def start_execution_cycle(self, step_id: int) -> ExecutionCycle:
        """새로운 실행 사이클을 시작합니다."""
        cycle = ExecutionCycle(step_id=step_id)
        
        if step_id not in self.execution["log"]:
            self.execution["log"][step_id] = []
        
        self.execution["log"][step_id].append(cycle)
        self.execution["total_cycles"] += 1
        
        return cycle
    
    def update_cycle_prediction(self, step_id: int, prediction: str) -> None:
        """현재 사이클의 예측을 업데이트합니다."""
        if step_id in self.execution["log"] and self.execution["log"][step_id]:
            self.execution["log"][step_id][-1].prediction = prediction
    
    def update_cycle_action(self, step_id: int, action: Dict[str, Any]) -> None:
        """현재 사이클의 행동을 업데이트합니다."""
        if step_id in self.execution["log"] and self.execution["log"][step_id]:
            self.execution["log"][step_id][-1].action = action
            self.execution["total_kg_queries"] += action.get("query_count", 0)
    
    def update_cycle_observation(self, step_id: int, observation: str) -> None:
        """현재 사이클의 관찰을 업데이트합니다."""
        if step_id in self.execution["log"] and self.execution["log"][step_id]:
            self.execution["log"][step_id][-1].observation = observation
    
    def update_cycle_thought(self, step_id: int, thought: str) -> None:
        """현재 사이클의 사고를 업데이트합니다."""
        if step_id in self.execution["log"] and self.execution["log"][step_id]:
            self.execution["log"][step_id][-1].thought = thought
    
    def update_step_status(self, step_id: int, status: str) -> None:
        """계획 단계의 상태를 업데이트합니다."""
        if step_id < len(self.strategy["overall_plan"]):
            self.strategy["overall_plan"][step_id].status = status
            if status == "InProgress":
                self.strategy["overall_plan"][step_id].attempts += 1
    
    def advance_to_next_step(self) -> bool:
        """다음 계획 단계로 진행합니다. 더 이상 단계가 없으면 False를 반환합니다."""
        self.execution["current_step_id"] += 1
        return self.execution["current_step_id"] < len(self.strategy["overall_plan"])
    
    def update_knowledge_graph(self, nodes: Dict[str, Any], edges: Dict[str, Any]) -> None:
        """지식 계층의 탐색 그래프를 업데이트합니다."""
        self.knowledge["exploration_graph"]["nodes"].update(nodes)
        self.knowledge["exploration_graph"]["edges"].update(edges)
    
    def add_candidate_entities(self, step_id: int, entities: List[str]) -> None:
        """특정 단계에서 발견된 후보 엔티티들을 추가합니다."""
        if step_id not in self.knowledge["candidate_entities"]:
            self.knowledge["candidate_entities"][step_id] = []
        self.knowledge["candidate_entities"][step_id].extend(entities)
    
    def add_reasoning_path(self, path: List[Dict[str, Any]]) -> None:
        """성공적인 추론 경로를 기록합니다."""
        self.knowledge["reasoning_paths"].append({
            "path": path,
            "timestamp": time.time(),
            "step_id": self.execution["current_step_id"]
        })
    
    def increment_llm_calls(self, count: int = 1) -> None:
        """LLM 호출 횟수를 증가시킵니다."""
        self.execution["total_llm_calls"] += count
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """메모리의 현재 상태를 요약하여 반환합니다."""
        return {
            "question": self.question,
            "status": self.strategy["status"],
            "current_step": self.execution["current_step_id"],
            "total_steps": len(self.strategy["overall_plan"]),
            "total_cycles": self.execution["total_cycles"],
            "total_llm_calls": self.execution["total_llm_calls"],
            "total_kg_queries": self.execution["total_kg_queries"],
            "replan_count": self.strategy["replan_count"],
            "entities_discovered": len(self.knowledge["exploration_graph"]["nodes"]),
            "reasoning_paths": len(self.knowledge["reasoning_paths"])
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """메모리 전체를 딕셔너리로 직렬화합니다."""
        return {
            "question": self.question,
            "created_at": self.created_at,
            "strategy": self.strategy,
            "execution": self.execution,
            "knowledge": self.knowledge
        }
    
    def save_to_file(self, filepath: str) -> None:
        """메모리를 JSON 파일로 저장합니다."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
