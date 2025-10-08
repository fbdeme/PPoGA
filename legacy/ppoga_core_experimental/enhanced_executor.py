"""
Enhanced Knowledge Graph Executor
Integrates PoG's proven KG functions with PPoGA's self-correction capabilities
"""

import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from ..pog_base.freebase_func import (
    entity_search,
    relation_search_prune,
    provide_triple,
    USE_MOCK,
)


@dataclass
class ExecutionResult:
    """Result of a KG execution step"""

    success: bool
    discovered_entities: List[str]
    relation_results: Dict[str, List[str]]
    observation: str
    execution_info: Dict[str, Any]
    error: Optional[str] = None


class EnhancedKGExecutor:
    """
    Enhanced KG Executor that combines PoG's proven functions
    with PPoGA's self-correction capabilities
    """

    def __init__(self, azure_config: Dict[str, Any]):
        self.azure_config = azure_config
        self.query_count = 0
        self.execution_history = []
        self.discovered_entities = set()

        # Display connection status
        if USE_MOCK:
            print("🔶 KG Executor: Using Mock Freebase (install Virtuoso for real KG)")
        else:
            print("🟢 KG Executor: Using Real Freebase connection")

    def get_available_relations(
        self,
        entity_id: str,
        entity_name: str,
        question: str,
        sub_questions: List[str] = None,
    ) -> List[str]:
        """
        Get available relations for an entity using PoG's relation_search_prune
        """
        print(f"🔍 Getting relations for: {entity_name}")

        try:
            # Use PoG's proven relation search
            relations_result, token_usage = relation_search_prune(
                entity_id=entity_id,
                sub_questions=sub_questions or [],
                entity_name=entity_name,
                pre_relations=[],
                pre_head=False,
                question=question,
                azure_config=self.azure_config,
            )

            # Extract relation names
            relation_names = [rel["relation"] for rel in relations_result]

            print(f"   Found {len(relation_names)} relations: {relation_names[:5]}...")
            return relation_names

        except Exception as e:
            print(f"❌ Error getting relations: {e}")
            return ["mock.relation.1", "mock.relation.2"]  # Fallback

    def execute_step_with_correction(
        self,
        entity_id: str,
        entity_name: str,
        selected_relations: List[str],
        step_description: str,
        prediction: Dict[str, Any] = None,
    ) -> ExecutionResult:
        """
        Execute KG step with self-correction capabilities
        """
        print(f"🤖 Executing: {step_description}")
        print(f"   Entity: {entity_name} ({entity_id})")
        print(f"   Relations: {selected_relations}")

        start_time = time.time()
        self.query_count += 1

        try:
            # Execute queries for each relation using PoG's proven functions
            all_discovered_entities = []
            relation_results = {}

            for relation in selected_relations:
                try:
                    # Use PoG's entity_search function
                    entity_list = entity_search(entity_id, relation, head=True)

                    # Process results using PoG's provide_triple
                    entity_names, entity_ids = provide_triple(entity_list, relation)

                    # Store results
                    relation_results[relation] = entity_names
                    all_discovered_entities.extend(entity_names)

                    print(f"     {relation}: {len(entity_names)} entities")

                except Exception as e:
                    print(f"⚠️ Error with relation {relation}: {e}")
                    relation_results[relation] = []

            # Create observation summary
            observation = self._create_observation_summary(
                entity_name, selected_relations, relation_results, step_description
            )

            # Self-correction: Evaluate against prediction
            success = self._evaluate_execution_success(
                prediction, relation_results, all_discovered_entities
            )

            execution_info = {
                "query_count": len(selected_relations),
                "execution_time": time.time() - start_time,
                "total_entities": len(all_discovered_entities),
                "unique_entities": len(set(all_discovered_entities)),
            }

            # Update discovered entities set
            self.discovered_entities.update(all_discovered_entities)

            result = ExecutionResult(
                success=success,
                discovered_entities=all_discovered_entities,
                relation_results=relation_results,
                observation=observation,
                execution_info=execution_info,
            )

            # Add to execution history
            self.execution_history.append(
                {
                    "step": step_description,
                    "entity": entity_name,
                    "relations": selected_relations,
                    "result": result,
                    "timestamp": time.time(),
                }
            )

            print(
                f"   ✅ Execution complete: {len(all_discovered_entities)} entities discovered"
            )
            return result

        except Exception as e:
            error_msg = f"Execution error: {e}"
            print(f"❌ {error_msg}")

            return ExecutionResult(
                success=False,
                discovered_entities=[],
                relation_results={},
                observation=f"Error during execution: {error_msg}",
                execution_info={
                    "query_count": 0,
                    "execution_time": time.time() - start_time,
                },
                error=error_msg,
            )

    def execute_with_retry(
        self,
        entity_id: str,
        entity_name: str,
        selected_relations: List[str],
        step_description: str,
        prediction: Dict[str, Any] = None,
        max_retries: int = 2,
    ) -> ExecutionResult:
        """
        Execute with retry logic for self-correction
        """
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"🔄 Retry attempt {attempt}/{max_retries}")
                # Modify relations for retry (self-correction)
                selected_relations = self._modify_relations_for_retry(
                    entity_id, entity_name, selected_relations, attempt
                )

            result = self.execute_step_with_correction(
                entity_id, entity_name, selected_relations, step_description, prediction
            )

            # Check if retry is needed
            if result.success or attempt == max_retries:
                return result

            print(f"   Attempt {attempt + 1} failed, retrying...")

        return result

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions"""
        return {
            "total_queries": self.query_count,
            "total_entities_discovered": len(self.discovered_entities),
            "execution_count": len(self.execution_history),
            "success_rate": sum(
                1 for h in self.execution_history if h["result"].success
            )
            / max(len(self.execution_history), 1),
            "recent_executions": (
                self.execution_history[-5:] if self.execution_history else []
            ),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return {
            "query_count": self.query_count,
            "unique_entities": len(self.discovered_entities),
            "execution_history_length": len(self.execution_history),
            "last_updated": time.time(),
            "connection_type": "Mock" if USE_MOCK else "Real Freebase",
        }

    # Helper methods

    def _create_observation_summary(
        self,
        entity_name: str,
        relations: List[str],
        results: Dict[str, List[str]],
        step_description: str,
    ) -> str:
        """Create human-readable observation summary"""
        total_entities = sum(len(entities) for entities in results.values())

        if total_entities == 0:
            return f"No entities found for {entity_name} using relations {relations}"

        summary_parts = [
            f"Explored {entity_name} and found {total_entities} total entities:"
        ]

        for relation, entities in results.items():
            if entities:
                entity_preview = entities[:3]
                more_text = (
                    f" (and {len(entities) - 3} more)" if len(entities) > 3 else ""
                )
                summary_parts.append(f"  - {relation}: {entity_preview}{more_text}")

        summary_parts.append(f"Step objective: {step_description}")

        return "\n".join(summary_parts)

    def _evaluate_execution_success(
        self,
        prediction: Dict[str, Any],
        relation_results: Dict[str, List[str]],
        discovered_entities: List[str],
    ) -> bool:
        """
        Evaluate if execution met predictions (self-correction evaluation)
        """
        if not prediction:
            # No prediction to compare against
            return len(discovered_entities) > 0

        # Check if we found expected entities
        expected_entities = prediction.get("expected_entities", [])
        if expected_entities:
            found_expected = any(
                any(
                    expected.lower() in entity.lower() for entity in discovered_entities
                )
                for expected in expected_entities
            )
            if found_expected:
                return True

        # Check if we used expected relations
        expected_relations = prediction.get("expected_relations", [])
        if expected_relations:
            used_expected = any(
                any(
                    expected.lower() in relation.lower()
                    for relation in relation_results.keys()
                )
                for expected in expected_relations
            )
            if used_expected:
                return True

        # Basic success: found any entities
        return len(discovered_entities) > 0

    def _modify_relations_for_retry(
        self,
        entity_id: str,
        entity_name: str,
        original_relations: List[str],
        attempt: int,
    ) -> List[str]:
        """
        Modify relations for retry attempts (self-correction strategy)
        """
        try:
            # Get fresh relations (might include different ones)
            all_available = self.get_available_relations(
                entity_id, entity_name, "retry query"
            )

            # Strategy 1: Try different relations
            if attempt == 1:
                # Remove failed relations and try new ones
                unused_relations = [
                    r for r in all_available if r not in original_relations
                ]
                if unused_relations:
                    return unused_relations[: len(original_relations)]

            # Strategy 2: Try broader set of relations
            if attempt == 2:
                # Use more relations
                return all_available[
                    : min(len(all_available), len(original_relations) + 2)
                ]

            # Fallback: return original
            return original_relations

        except Exception as e:
            print(f"⚠️ Error modifying relations for retry: {e}")
            return original_relations


class MockKGExecutor:
    """Simple mock executor for testing without KG"""

    def __init__(self, azure_config: Dict[str, Any]):
        self.azure_config = azure_config
        self.query_count = 0
        print("🔶 Using Mock KG Executor")

    def get_available_relations(
        self,
        entity_id: str,
        entity_name: str,
        question: str,
        sub_questions: List[str] = None,
    ) -> List[str]:
        """Mock relation discovery"""
        mock_relations = [
            "music.artist.album",
            "music.artist.track",
            "award.award_winner.awards_won",
            "people.person.profession",
            "location.location.contains",
        ]
        return mock_relations[:3]  # Return subset

    def execute_step_with_correction(
        self,
        entity_id: str,
        entity_name: str,
        selected_relations: List[str],
        step_description: str,
        prediction: Dict[str, Any] = None,
    ) -> ExecutionResult:
        """Mock execution"""
        self.query_count += 1

        # Mock discovered entities based on entity name
        mock_entities = []
        if "taylor" in entity_name.lower():
            mock_entities = ["1989", "Folklore", "Shake It Off", "Grammy Award"]
        elif "music" in str(selected_relations).lower():
            mock_entities = ["Album 1", "Song 1", "Artist 1"]
        else:
            mock_entities = [f"Mock_Entity_{i}" for i in range(3)]

        relation_results = {rel: mock_entities[:2] for rel in selected_relations}

        observation = (
            f"Mock execution for {entity_name}: found {len(mock_entities)} entities"
        )

        return ExecutionResult(
            success=True,
            discovered_entities=mock_entities,
            relation_results=relation_results,
            observation=observation,
            execution_info={
                "query_count": len(selected_relations),
                "execution_time": 0.1,
            },
        )

    def execute_with_retry(
        self,
        entity_id: str,
        entity_name: str,
        selected_relations: List[str],
        step_description: str,
        prediction: Dict[str, Any] = None,
        max_retries: int = 2,
    ) -> ExecutionResult:
        """Mock retry execution"""
        return self.execute_step_with_correction(
            entity_id, entity_name, selected_relations, step_description, prediction
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Mock statistics"""
        return {
            "query_count": self.query_count,
            "connection_type": "Mock",
            "last_updated": time.time(),
        }
