"""
PPoGA Executor Module

PPoGA의 실행가(Executor) 역할을 담당하는 모듈입니다.
실제 지식 그래프와의 상호작용을 수행합니다.
"""

import sys
import os
from typing import Dict, List, Any, Tuple

# PoG 모듈 import를 위한 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'PoG'))

from ppoga_memory import PPoGAMemory

try:
    from freebase_func import entity_search, relation_search_prune
    from utils import run_llm
except ImportError:
    print("Warning: Could not import PoG modules. Mock functions will be used.")
    
    def entity_search(entity_id, relation, is_head=True):
        """Mock function for entity search"""
        return [f"mock_entity_{i}" for i in range(3)]
    
    def relation_search_prune(entity_id, sub_questions, entity_name, pre_relations, pre_head, question, args):
        """Mock function for relation search"""
        return [{"relation": "mock.relation", "entity": entity_id, "head": True}], {"total": 0, "input": 0, "output": 0}


class PPoGAExecutor:
    """
    PPoGA의 실행가(Executor) 클래스
    
    주요 역할:
    1. 지식 그래프 쿼리 실행
    2. 결과 정제 및 요약
    3. Planner와의 협업
    """
    
    def __init__(self, kg_config: Dict[str, Any]):
        """
        Executor 초기화
        
        Args:
            kg_config: 지식 그래프 설정
        """
        self.kg_config = kg_config
        self.query_count = 0
    
    def execute_step(self, entity_id: str, entity_name: str, selected_relations: List[str], 
                    current_step_description: str, memory: PPoGAMemory) -> Tuple[str, Dict[str, Any]]:
        """
        계획 단계를 실행하여 지식 그래프에서 정보를 가져옵니다.
        
        Args:
            entity_id: 탐색할 엔티티 ID
            entity_name: 엔티티 이름
            selected_relations: Planner가 선택한 관계 목록
            current_step_description: 현재 계획 단계 설명
            memory: PPoGA 메모리 객체
            
        Returns:
            Tuple[str, Dict[str, Any]]: (관찰 결과, 실행 정보)
        """
        print(f"🤖 Executor: {entity_name}에 대해 {len(selected_relations)}개 관계 탐색 중...")
        
        all_results = {}
        total_entities_found = 0
        execution_info = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "relations_explored": selected_relations,
            "query_count": 0,
            "sparql_queries": []
        }
        
        # 각 선택된 관계에 대해 탐색 수행
        for relation in selected_relations:
            try:
                print(f"   🔍 관계 '{relation}' 탐색 중...")
                
                # PoG의 entity_search 함수 사용
                found_entities = entity_search(entity_id, relation, True)  # head=True로 가정
                
                if found_entities:
                    all_results[relation] = found_entities
                    total_entities_found += len(found_entities)
                    print(f"      ✅ {len(found_entities)}개 엔티티 발견")
                else:
                    print(f"      ❌ 엔티티 없음")
                
                execution_info["query_count"] += 1
                execution_info["sparql_queries"].append({
                    "relation": relation,
                    "entity": entity_id,
                    "result_count": len(found_entities) if found_entities else 0
                })
                
            except Exception as e:
                print(f"      ⚠️ 관계 '{relation}' 탐색 오류: {e}")
                all_results[relation] = []
        
        # 결과 요약 생성
        observation = self._create_observation_summary(
            entity_name, selected_relations, all_results, current_step_description
        )
        
        # 메모리 업데이트
        self._update_memory_with_results(memory, entity_id, all_results, execution_info)
        
        self.query_count += execution_info["query_count"]
        
        print(f"✅ Executor: 실행 완료 - 총 {total_entities_found}개 엔티티 발견")
        
        return observation, execution_info
    
    def _create_observation_summary(self, entity_name: str, relations: List[str], 
                                  results: Dict[str, List[str]], step_description: str) -> str:
        """
        실행 결과를 Planner가 이해하기 쉬운 형태로 요약합니다.
        
        Args:
            entity_name: 엔티티 이름
            relations: 탐색한 관계들
            results: 관계별 결과
            step_description: 현재 단계 설명
            
        Returns:
            str: 요약된 관찰 결과
        """
        total_entities = sum(len(entities) for entities in results.values())
        
        if total_entities == 0:
            return f"{entity_name}에서 {len(relations)}개 관계를 탐색했지만 관련 엔티티를 찾지 못했습니다."
        
        summary_parts = [f"{entity_name}에서 총 {total_entities}개의 관련 엔티티를 발견했습니다."]
        
        for relation, entities in results.items():
            if entities:
                entity_sample = entities[:3]  # 처음 3개만 표시
                more_text = f" 외 {len(entities)-3}개" if len(entities) > 3 else ""
                summary_parts.append(f"- {relation}: {', '.join(entity_sample)}{more_text}")
        
        return " ".join(summary_parts)
    
    def _update_memory_with_results(self, memory: PPoGAMemory, entity_id: str, 
                                  results: Dict[str, List[str]], execution_info: Dict[str, Any]) -> None:
        """
        실행 결과를 메모리에 업데이트합니다.
        
        Args:
            memory: PPoGA 메모리 객체
            entity_id: 탐색한 엔티티 ID
            results: 탐색 결과
            execution_info: 실행 정보
        """
        # 지식 그래프 노드 및 엣지 업데이트
        nodes = {}
        edges = {}
        
        # 현재 엔티티를 노드로 추가
        nodes[entity_id] = {
            "name": execution_info["entity_name"],
            "explored": True,
            "timestamp": execution_info.get("timestamp", "")
        }
        
        # 발견된 엔티티들을 노드와 엣지로 추가
        for relation, entities in results.items():
            for entity in entities:
                # 새로운 엔티티를 노드로 추가
                nodes[entity] = {
                    "name": entity,
                    "explored": False,
                    "discovered_via": relation
                }
                
                # 엣지 추가
                edge_key = f"{entity_id}-{relation}-{entity}"
                edges[edge_key] = {
                    "source": entity_id,
                    "relation": relation,
                    "target": entity,
                    "timestamp": execution_info.get("timestamp", "")
                }
        
        memory.update_knowledge_graph(nodes, edges)
        
        # 후보 엔티티 추가
        current_step_id = memory.execution["current_step_id"]
        all_discovered_entities = []
        for entities in results.values():
            all_discovered_entities.extend(entities)
        
        if all_discovered_entities:
            memory.add_candidate_entities(current_step_id, all_discovered_entities)
    
    def get_available_relations(self, entity_id: str, entity_name: str, 
                              question: str, sub_questions: List[str]) -> List[str]:
        """
        엔티티에 대해 사용 가능한 관계 목록을 가져옵니다.
        
        Args:
            entity_id: 엔티티 ID
            entity_name: 엔티티 이름
            question: 원래 질문
            sub_questions: 하위 질문들
            
        Returns:
            List[str]: 사용 가능한 관계 목록
        """
        try:
            # PoG의 relation_search_prune 함수 사용
            # 임시 args 객체 생성
            class TempArgs:
                def __init__(self):
                    self.LLM_type = "gpt-3.5-turbo"
                    self.opeani_api_keys = ""
                    self.temperature_exploration = 0.3
                    self.max_length = 2048
            
            args = TempArgs()
            
            relations_data, token_info = relation_search_prune(
                entity_id, sub_questions, entity_name, [], -1, question, args
            )
            
            # 관계 목록 추출
            relations = [rel_data["relation"] for rel_data in relations_data if "relation" in rel_data]
            
            print(f"🔍 Executor: {entity_name}에 대해 {len(relations)}개 관계 발견")
            
            return relations
            
        except Exception as e:
            print(f"⚠️ Executor: 관계 탐색 오류: {e}")
            # 기본 관계 목록 반환
            return ["type.object.type", "common.topic.notable_types"]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        실행 통계를 반환합니다.
        
        Returns:
            Dict[str, Any]: 실행 통계
        """
        return {
            "total_queries": self.query_count,
            "kg_config": self.kg_config
        }
