import sys
import os
from typing import Dict, List, Any, Tuple

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PoG.freebase_func import relation_search_prune, entity_search, provide_triple


class EnhancedExecutor:
    """
    Enhanced Executor that connects PPoGA planner with PoG's proven SPARQL engine

    This executor bridges the gap between strategic planning and actual knowledge graph execution
    """

    def __init__(self, kg_config: Dict[str, Any]):
        self.kg_config = kg_config
        self.query_count = 0
        self.total_entities_discovered = 0

        # Statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "entities_discovered": 0,
            "relations_explored": 0,
        }

    def execute_step(
        self, entity_id: str, entity_name: str, plan_step: Dict[str, Any], args: Any
    ) -> Dict[str, Any]:
        """
        Execute a plan step by exploring the knowledge graph

        This method uses PoG's proven SPARQL functions to perform actual KG queries
        """
        print(f"🤖 Enhanced Executor: Executing step for entity {entity_name}")

        try:
            # Use PoG's relation search with pruning
            sub_questions = [plan_step.get("objective", "")]
            pre_relations = []
            pre_head = False

            retrieved_relations, token_info = relation_search_prune(
                entity_id=entity_id,
                sub_questions=sub_questions,
                entity_name=entity_name,
                pre_relations=pre_relations,
                pre_head=pre_head,
                question=plan_step["description"],
                args=args,
            )

            self.stats["relations_explored"] += len(retrieved_relations)

            # Execute entity search for each selected relation
            discovered_entities = {}
            exploration_results = {}

            for relation_info in retrieved_relations:
                relation = relation_info["relation"]
                is_head = relation_info["head"]

                # Use PoG's entity search
                entity_candidates_id = entity_search(entity_id, relation, is_head)

                if entity_candidates_id:
                    # Convert IDs to names
                    entity_candidates, entity_candidates_id = provide_triple(
                        entity_candidates_id, relation
                    )

                    # Store results
                    exploration_results[relation] = {
                        "entity_ids": entity_candidates_id,
                        "entity_names": entity_candidates,
                        "is_head": is_head,
                    }

                    # Update discovered entities
                    for i, ent_id in enumerate(entity_candidates_id):
                        if i < len(entity_candidates):
                            discovered_entities[ent_id] = entity_candidates[i]

                self.stats["total_queries"] += 1

            # Create observation summary
            observation = self._create_observation_summary(
                entity_name, exploration_results, plan_step
            )

            self.stats["successful_queries"] += 1
            self.stats["entities_discovered"] += len(discovered_entities)

            return {
                "success": True,
                "observation": observation,
                "discovered_entities": discovered_entities,
                "exploration_results": exploration_results,
                "relations_explored": list(exploration_results.keys()),
                "execution_info": {
                    "query_count": len(retrieved_relations),
                    "entities_found": len(discovered_entities),
                    "relations_found": len(exploration_results),
                },
            }

        except Exception as e:
            print(f"❌ Execution error: {e}")
            self.stats["failed_queries"] += 1

            return {
                "success": False,
                "observation": f"Execution failed: {str(e)}",
                "discovered_entities": {},
                "exploration_results": {},
                "relations_explored": [],
                "execution_info": {
                    "query_count": 0,
                    "entities_found": 0,
                    "relations_found": 0,
                },
                "error": str(e),
            }

    def get_available_relations(
        self, entity_id: str, entity_name: str, args: Any
    ) -> List[str]:
        """Get available relations for an entity (lightweight version)"""
        try:
            # Use a simplified version to just get relation names
            retrieved_relations, _ = relation_search_prune(
                entity_id=entity_id,
                sub_questions=["explore relations"],
                entity_name=entity_name,
                pre_relations=[],
                pre_head=False,
                question="get available relations",
                args=args,
            )

            return [rel["relation"] for rel in retrieved_relations]

        except Exception as e:
            print(f"❌ Error getting relations: {e}")
            return []

    def select_relations(
        self,
        entity_id: str,
        entity_name: str,
        available_relations: List[str],
        plan_step: Dict[str, Any],
        question: str,
    ) -> List[str]:
        """
        Select most relevant relations for the current plan step

        This could be enhanced with LLM-based selection logic
        """
        # For now, use simple heuristics
        # In a full implementation, this would use LLM to select relations
        # based on the plan step objective and question

        # Priority keywords based on common question patterns
        priority_keywords = {
            "who": ["person", "people", "actor", "director", "president", "author"],
            "what": ["type", "name", "title", "profession"],
            "when": ["date", "time", "year", "born", "died"],
            "where": ["location", "place", "country", "city", "lived"],
            "how": ["method", "way", "cause", "reason"],
        }

        question_lower = question.lower()
        step_description = plan_step.get("description", "").lower()

        # Score relations based on relevance
        scored_relations = []
        for relation in available_relations:
            score = 0
            relation_lower = relation.lower()

            # Check for question type matches
            for q_type, keywords in priority_keywords.items():
                if q_type in question_lower:
                    for keyword in keywords:
                        if keyword in relation_lower:
                            score += 2

            # Check for step description matches
            step_words = step_description.split()
            for word in step_words:
                if len(word) > 3 and word in relation_lower:
                    score += 1

            scored_relations.append((relation, score))

        # Sort by score and return top relations
        scored_relations.sort(key=lambda x: x[1], reverse=True)
        selected = [rel for rel, score in scored_relations[:5]]  # Top 5 relations

        print(
            f"   Selected {len(selected)} relations from {len(available_relations)} available"
        )
        return selected

    def _create_observation_summary(
        self,
        entity_name: str,
        exploration_results: Dict[str, Any],
        plan_step: Dict[str, Any],
    ) -> str:
        """Create a human-readable summary of exploration results"""
        if not exploration_results:
            return f"No information found for {entity_name}"

        summary_parts = [f"Exploration results for {entity_name}:"]

        for relation, results in exploration_results.items():
            entity_names = results["entity_names"]
            entity_count = len(entity_names)

            if entity_count > 0:
                if entity_count <= 3:
                    entities_str = ", ".join(entity_names)
                else:
                    entities_str = (
                        f"{', '.join(entity_names[:3])} and {entity_count - 3} more"
                    )

                summary_parts.append(f"  - {relation}: {entities_str}")
            else:
                summary_parts.append(f"  - {relation}: No entities found")

        total_entities = sum(
            len(results["entity_names"]) for results in exploration_results.values()
        )
        summary_parts.append(f"Total entities discovered: {total_entities}")

        return "\n".join(summary_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return self.stats.copy()
