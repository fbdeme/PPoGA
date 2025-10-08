import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ExecutionCycle:
    """Single execution cycle (Prediction-Action-Observation-Thought)"""

    step_id: int
    prediction: Dict[str, Any]
    action: Dict[str, Any]
    observation: str
    thought: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanStep:
    """Single step in the strategic plan"""

    step_id: int
    description: str
    objective: str
    expected_outcome: str
    status: str = "not_started"  # not_started, in_progress, completed, failed


class PPoGAMemory:
    """
    Enhanced 3-layer memory system for PPoGA

    Strategic Layer: Plans, rationale, and strategic decisions
    Execution Layer: Prediction-Action-Observation-Thought cycles
    Knowledge Layer: Discovered entities, relations, and reasoning paths
    """

    def __init__(self, question: str):
        self.question = question
        self.created_at = time.time()

        # Strategic Layer: "Where are we going?"
        self.strategy = {
            "initial_question": question,
            "overall_plan": [],  # List[PlanStep]
            "plan_rationale": "",
            "status": "Planning",  # Planning, Executing, Evaluating, Replanning, Finished
            "alternative_plans": [],  # Previous or alternative strategic plans
            "replan_count": 0,
        }

        # Execution Layer: "How are we proceeding?"
        self.execution = {
            "current_step_id": 0,
            "cycles": {},  # Key: step_id, Value: List[ExecutionCycle]
            "total_cycles": 0,
            "current_prediction": {},
            "current_observation": "",
            "current_thought": {},
        }

        # Knowledge Layer: "What have we learned?"
        self.knowledge = {
            "discovered_entities": {},  # entity_id -> entity_name
            "explored_relations": {},  # entity_id -> List[relation]
            "exploration_graph": {},  # entity_id -> {relation -> [target_entities]}
            "reasoning_chains": [],  # List of reasoning paths
            "candidates": [],  # Current answer candidates
        }

        # Statistics
        self.stats = {
            "llm_calls": 0,
            "sparql_queries": 0,
            "total_entities_explored": 0,
            "total_relations_explored": 0,
            "execution_time": 0,
        }

    def update_plan(self, plan_steps: List[Dict], rationale: str):
        """Update the strategic plan"""
        self.strategy["overall_plan"] = [
            PlanStep(
                step_id=step["step_id"],
                description=step["description"],
                objective=step.get("objective", ""),
                expected_outcome=step.get("expected_outcome", ""),
            )
            for step in plan_steps
        ]
        self.strategy["plan_rationale"] = rationale
        self.strategy["status"] = "Executing"
        self.execution["current_step_id"] = 0

    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step in the plan"""
        if self.execution["current_step_id"] < len(self.strategy["overall_plan"]):
            return self.strategy["overall_plan"][self.execution["current_step_id"]]
        return None

    def advance_step(self) -> bool:
        """Move to the next step"""
        current_step = self.get_current_step()
        if current_step:
            current_step.status = "completed"

        self.execution["current_step_id"] += 1
        return self.execution["current_step_id"] < len(self.strategy["overall_plan"])

    def add_execution_cycle(
        self,
        step_id: int,
        prediction: Dict,
        action: Dict,
        observation: str,
        thought: Dict,
    ):
        """Add a complete execution cycle"""
        if step_id not in self.execution["cycles"]:
            self.execution["cycles"][step_id] = []

        cycle = ExecutionCycle(
            step_id=step_id,
            prediction=prediction,
            action=action,
            observation=observation,
            thought=thought,
        )

        self.execution["cycles"][step_id].append(cycle)
        self.execution["total_cycles"] += 1

        # Update current state
        self.execution["current_prediction"] = prediction
        self.execution["current_observation"] = observation
        self.execution["current_thought"] = thought

    def add_discovered_entities(self, entities: Dict[str, str]):
        """Add newly discovered entities"""
        self.knowledge["discovered_entities"].update(entities)
        self.stats["total_entities_explored"] = len(
            self.knowledge["discovered_entities"]
        )

    def add_exploration_result(
        self, entity_id: str, relation: str, target_entities: List[str]
    ):
        """Add exploration results to the knowledge graph"""
        if entity_id not in self.knowledge["exploration_graph"]:
            self.knowledge["exploration_graph"][entity_id] = {}

        self.knowledge["exploration_graph"][entity_id][relation] = target_entities

        # Update explored relations
        if entity_id not in self.knowledge["explored_relations"]:
            self.knowledge["explored_relations"][entity_id] = []

        if relation not in self.knowledge["explored_relations"][entity_id]:
            self.knowledge["explored_relations"][entity_id].append(relation)
            self.stats["total_relations_explored"] += 1

    def add_reasoning_chain(self, chain: List[Tuple[str, str, str]]):
        """Add a reasoning chain (list of (entity, relation, entity) triplets)"""
        self.knowledge["reasoning_chains"].append(chain)

    def replan(self, new_plan: List[Dict], new_rationale: str, reason: str):
        """Start replanning with a new strategy"""
        # Archive current plan
        self.strategy["alternative_plans"].append(
            {
                "plan": self.strategy["overall_plan"],
                "rationale": self.strategy["plan_rationale"],
                "abandoned_at": time.time(),
                "reason": reason,
            }
        )

        # Set new plan
        self.update_plan(new_plan, new_rationale)
        self.strategy["replan_count"] += 1
        self.strategy["status"] = "Replanning"

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current memory state"""
        current_step = self.get_current_step()

        return {
            "question": self.question,
            "current_step": current_step.description if current_step else "Completed",
            "step_progress": f"{self.execution['current_step_id']}/{len(self.strategy['overall_plan'])}",
            "total_steps": len(self.strategy["overall_plan"]),
            "entities_discovered": len(self.knowledge["discovered_entities"]),
            "relations_explored": self.stats["total_relations_explored"],
            "execution_cycles": self.execution["total_cycles"],
            "replan_count": self.strategy["replan_count"],
            "llm_calls": self.stats["llm_calls"],
            "sparql_queries": self.stats["sparql_queries"],
            "status": self.strategy["status"],
            "reasoning_chains": len(self.knowledge["reasoning_chains"]),
        }

    def get_context_for_llm(self) -> str:
        """Get formatted context for LLM prompts"""
        summary = self.get_memory_summary()
        current_step = self.get_current_step()

        context = f"Current Question: {self.question}\n"
        context += (
            f"Progress: Step {summary['step_progress']} - {summary['current_step']}\n"
        )
        context += f"Discovered Entities: {summary['entities_discovered']}\n"
        context += f"Explored Relations: {summary['relations_explored']}\n"

        if self.knowledge["reasoning_chains"]:
            context += f"Recent Reasoning Chains:\n"
            for i, chain in enumerate(
                self.knowledge["reasoning_chains"][-3:]
            ):  # Last 3 chains
                context += (
                    f"  {i+1}. {' -> '.join([f'{e}({r})' for e, r, _ in chain])}\n"
                )

        return context

    def increment_llm_calls(self):
        """Increment LLM call counter"""
        self.stats["llm_calls"] += 1

    def increment_sparql_queries(self):
        """Increment SPARQL query counter"""
        self.stats["sparql_queries"] += 1
