"""
PPoGA Enhanced Memory System
3-Layer Memory Architecture: Strategic, Execution, Knowledge
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class PlanStatus(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    REPLANNING = "replanning"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class ExecutionCycle:
    """Single Prediction-Action-Observation-Thought cycle"""

    step_id: int
    prediction: Dict[str, Any]
    action: str
    observation: str
    thought: Dict[str, Any]
    timestamp: float
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanStep:
    """Enhanced plan step with prediction capabilities"""

    step_id: int
    description: str
    objective: str
    expected_entities: List[str] = field(default_factory=list)
    predicted_outcome: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    status: str = "not_started"  # not_started, in_progress, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplorationPath:
    """Knowledge exploration path in the graph"""

    start_entity: str
    relation: str
    end_entities: List[str]
    step_id: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ThreeLayerMemory:
    """
    PPoGA's 3-Layer Memory Architecture

    Strategic Layer: WHY - Plans, rationale, strategy changes
    Execution Layer: HOW - Prediction-Action-Observation-Thought cycles
    Knowledge Layer: WHAT - Discovered entities, reasoning chains, exploration graph
    """

    def __init__(self, question: str):
        self.question = question
        self.created_at = time.time()
        self.last_updated = time.time()

        # =================================================================
        # STRATEGIC LAYER: "Where are we going and why?"
        # =================================================================
        self.strategic = {
            "initial_question": question,
            "current_plan": [],  # List of PlanStep objects
            "plan_rationale": "",
            "strategy": "",
            "status": PlanStatus.PLANNING.value,
            "alternative_plans": [],  # Previous/failed plans
            "replan_count": 0,
            "replan_history": [],  # History of why replanning occurred
            "overall_confidence": 0.5,
        }

        # =================================================================
        # EXECUTION LAYER: "How are we progressing?"
        # =================================================================
        self.execution = {
            "current_step_id": 0,
            "execution_cycles": [],  # List of ExecutionCycle objects
            "total_cycles": 0,
            "successful_cycles": 0,
            "failed_cycles": 0,
            "current_prediction": {},
            "current_observation": "",
            "current_thought": {},
            "correction_attempts": [],  # Track self-correction attempts
            "performance_metrics": {
                "prediction_accuracy": [],
                "step_success_rate": 0.0,
                "avg_confidence": 0.0,
            },
        }

        # =================================================================
        # KNOWLEDGE LAYER: "What have we learned?"
        # =================================================================
        self.knowledge = {
            "exploration_graph": {},  # Entity -> Relations -> Entities
            "discovered_entities": [],  # All discovered entities
            "entity_details": {},  # Entity ID -> Name mappings
            "reasoning_chains": [],  # Chains of reasoning
            "exploration_paths": [],  # List of ExplorationPath objects
            "key_findings": [],  # Important discoveries
            "entity_relevance_scores": {},  # Entity -> relevance score
            "confidence_scores": {},  # Track confidence over time
        }

        # =================================================================
        # STATISTICS & METADATA
        # =================================================================
        self.statistics = {
            "llm_calls": 0,
            "kg_queries": 0,
            "total_execution_time": 0.0,
            "memory_updates": 0,
            "peak_entities": 0,
        }

    # =================================================================
    # STRATEGIC LAYER METHODS
    # =================================================================

    def update_plan(
        self, plan_steps: List[Dict[str, Any]], rationale: str, strategy: str = ""
    ):
        """Update the current strategic plan"""
        # Convert dict plan steps to PlanStep objects
        plan_objects = []
        for step_dict in plan_steps:
            plan_step = PlanStep(
                step_id=step_dict.get("step_id", len(plan_objects) + 1),
                description=step_dict.get("description", ""),
                objective=step_dict.get("objective", ""),
                expected_entities=step_dict.get("expected_entities", []),
                predicted_outcome=step_dict.get("predicted_outcome", {}),
                confidence=step_dict.get("confidence", 0.5),
            )
            plan_objects.append(plan_step)

        # Archive current plan if exists
        if self.strategic["current_plan"]:
            self.strategic["alternative_plans"].append(
                {
                    "plan": [step.to_dict() for step in self.strategic["current_plan"]],
                    "rationale": self.strategic["plan_rationale"],
                    "timestamp": time.time(),
                    "reason_archived": "Updated with new plan",
                }
            )

        # Update strategic layer
        self.strategic["current_plan"] = plan_objects
        self.strategic["plan_rationale"] = rationale
        self.strategic["strategy"] = strategy
        self.strategic["status"] = PlanStatus.EXECUTING.value
        self.execution["current_step_id"] = 0

        self._update_timestamp()
        print(f"📚 Memory: Plan updated with {len(plan_objects)} steps")

    def replan(self, new_plan: List[Dict[str, Any]], reason: str, strategy_change: str):
        """Record a replanning event"""
        self.strategic["replan_count"] += 1
        self.strategic["replan_history"].append(
            {
                "replan_id": self.strategic["replan_count"],
                "reason": reason,
                "strategy_change": strategy_change,
                "timestamp": time.time(),
                "previous_step": self.execution["current_step_id"],
            }
        )

        # Update with new plan
        self.update_plan(
            new_plan,
            f"Replan #{self.strategic['replan_count']}: {reason}",
            strategy_change,
        )
        print(f"🔄 Memory: Replanning #{self.strategic['replan_count']}")

    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get current plan step"""
        current_id = self.execution["current_step_id"]
        if current_id < len(self.strategic["current_plan"]):
            return self.strategic["current_plan"][current_id].to_dict()
        return None

    def advance_step(self) -> bool:
        """Advance to next step"""
        if self.execution["current_step_id"] < len(self.strategic["current_plan"]) - 1:
            self.execution["current_step_id"] += 1
            self._update_timestamp()
            return True
        else:
            self.strategic["status"] = PlanStatus.FINISHED.value
            return False

    # =================================================================
    # EXECUTION LAYER METHODS
    # =================================================================

    def add_execution_cycle(
        self,
        step_id: int,
        prediction: Dict[str, Any],
        action: str,
        observation: str,
        thought: Dict[str, Any],
        success: bool = True,
    ):
        """Add a complete Prediction-Action-Observation-Thought cycle"""
        cycle = ExecutionCycle(
            step_id=step_id,
            prediction=prediction,
            action=action,
            observation=observation,
            thought=thought,
            timestamp=time.time(),
            success=success,
        )

        self.execution["execution_cycles"].append(cycle)
        self.execution["total_cycles"] += 1

        if success:
            self.execution["successful_cycles"] += 1
        else:
            self.execution["failed_cycles"] += 1

        # Update current state
        self.execution["current_prediction"] = prediction
        self.execution["current_observation"] = observation
        self.execution["current_thought"] = thought

        # Update performance metrics
        self._update_performance_metrics(cycle)

        self._update_timestamp()
        print(f"📝 Memory: Added execution cycle for step {step_id}")

    def add_correction_attempt(
        self, step_id: int, attempt_type: str, reason: str, result: str
    ):
        """Record a self-correction attempt"""
        correction = {
            "step_id": step_id,
            "attempt_type": attempt_type,  # "path_correction", "replan", etc.
            "reason": reason,
            "result": result,
            "timestamp": time.time(),
        }

        self.execution["correction_attempts"].append(correction)
        self._update_timestamp()
        print(f"🔧 Memory: Recorded correction attempt for step {step_id}")

    # =================================================================
    # KNOWLEDGE LAYER METHODS
    # =================================================================

    def add_exploration_path(
        self, start_entity: str, relation: str, end_entities: List[str], step_id: int
    ):
        """Add knowledge exploration path"""
        path = ExplorationPath(
            start_entity=start_entity,
            relation=relation,
            end_entities=end_entities,
            step_id=step_id,
            timestamp=time.time(),
        )

        self.knowledge["exploration_paths"].append(path)

        # Update exploration graph
        if start_entity not in self.knowledge["exploration_graph"]:
            self.knowledge["exploration_graph"][start_entity] = {}

        if relation not in self.knowledge["exploration_graph"][start_entity]:
            self.knowledge["exploration_graph"][start_entity][relation] = []

        self.knowledge["exploration_graph"][start_entity][relation].extend(end_entities)

        # Update discovered entities
        all_entities = [start_entity] + end_entities
        for entity in all_entities:
            if entity not in self.knowledge["discovered_entities"]:
                self.knowledge["discovered_entities"].append(entity)

        self._update_statistics()
        self._update_timestamp()
        print(
            f"🗺️ Memory: Added exploration path {start_entity} -> {relation} -> {len(end_entities)} entities"
        )

    def add_key_finding(
        self, finding: str, evidence: List[str], confidence: float, step_id: int
    ):
        """Add important discovery to knowledge"""
        key_finding = {
            "finding": finding,
            "evidence": evidence,
            "confidence": confidence,
            "step_id": step_id,
            "timestamp": time.time(),
        }

        self.knowledge["key_findings"].append(key_finding)
        self._update_timestamp()
        print(f"💡 Memory: Added key finding: {finding[:50]}...")

    def update_entity_relevance(self, entity: str, relevance_score: float, reason: str):
        """Update entity relevance score"""
        self.knowledge["entity_relevance_scores"][entity] = {
            "score": relevance_score,
            "reason": reason,
            "timestamp": time.time(),
        }
        self._update_timestamp()

    # =================================================================
    # MEMORY SUMMARY AND ANALYSIS
    # =================================================================

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary"""
        return {
            # Question and basic info
            "question": self.question,
            "status": self.strategic["status"],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            # Strategic summary
            "strategic_summary": {
                "current_step": self.execution["current_step_id"] + 1,
                "total_steps": len(self.strategic["current_plan"]),
                "replan_count": self.strategic["replan_count"],
                "overall_strategy": self.strategic["strategy"],
                "plan_confidence": self.strategic["overall_confidence"],
            },
            # Execution summary
            "execution_summary": {
                "total_cycles": self.execution["total_cycles"],
                "successful_cycles": self.execution["successful_cycles"],
                "success_rate": self.execution["successful_cycles"]
                / max(self.execution["total_cycles"], 1),
                "correction_attempts": len(self.execution["correction_attempts"]),
                "avg_confidence": self.execution["performance_metrics"][
                    "avg_confidence"
                ],
            },
            # Knowledge summary
            "knowledge_summary": {
                "entities_discovered": len(self.knowledge["discovered_entities"]),
                "exploration_paths": len(self.knowledge["exploration_paths"]),
                "key_findings": len(self.knowledge["key_findings"]),
                "graph_density": len(self.knowledge["exploration_graph"]),
            },
            # Statistics
            "statistics": self.statistics.copy(),
        }

    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Extract reasoning chain from execution cycles"""
        reasoning_chain = []

        for cycle in self.execution["execution_cycles"]:
            reasoning_step = {
                "step_id": cycle.step_id,
                "prediction": cycle.prediction.get("summary", "No prediction"),
                "action": cycle.action,
                "observation": (
                    cycle.observation[:200] + "..."
                    if len(cycle.observation) > 200
                    else cycle.observation
                ),
                "thought": cycle.thought.get("reasoning", "No reasoning"),
                "confidence": cycle.thought.get("confidence", 0.5),
                "success": cycle.success,
            }
            reasoning_chain.append(reasoning_step)

        return reasoning_chain

    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge exploration"""
        return {
            "total_entities": len(self.knowledge["discovered_entities"]),
            "exploration_graph": self.knowledge["exploration_graph"],
            "recent_paths": [
                path.to_dict() for path in self.knowledge["exploration_paths"][-5:]
            ],
            "key_findings": self.knowledge["key_findings"],
            "entity_relevance": self.knowledge["entity_relevance_scores"],
        }

    # =================================================================
    # STATISTICS AND MAINTENANCE
    # =================================================================

    def increment_llm_calls(self):
        """Increment LLM call counter"""
        self.statistics["llm_calls"] += 1
        self._update_timestamp()

    def increment_kg_queries(self, count: int = 1):
        """Increment KG query counter"""
        self.statistics["kg_queries"] += count
        self._update_timestamp()

    def add_execution_time(self, time_seconds: float):
        """Add execution time"""
        self.statistics["total_execution_time"] += time_seconds
        self._update_timestamp()

    def _update_performance_metrics(self, cycle: ExecutionCycle):
        """Update execution performance metrics"""
        # Track prediction accuracy
        if "confidence" in cycle.prediction:
            self.execution["performance_metrics"]["prediction_accuracy"].append(
                cycle.prediction["confidence"]
            )

        # Update success rate
        total = self.execution["total_cycles"]
        success = self.execution["successful_cycles"]
        self.execution["performance_metrics"]["step_success_rate"] = success / max(
            total, 1
        )

        # Update average confidence
        if cycle.thought and "confidence" in cycle.thought:
            confidences = [
                c
                for c in self.execution["performance_metrics"]["prediction_accuracy"]
                if c > 0
            ]
            if confidences:
                self.execution["performance_metrics"]["avg_confidence"] = sum(
                    confidences
                ) / len(confidences)

    def _update_statistics(self):
        """Update general statistics"""
        self.statistics["memory_updates"] += 1
        current_entities = len(self.knowledge["discovered_entities"])
        if current_entities > self.statistics["peak_entities"]:
            self.statistics["peak_entities"] = current_entities

    def _update_timestamp(self):
        """Update last modified timestamp"""
        self.last_updated = time.time()

    # =================================================================
    # DEBUG AND EXPORT
    # =================================================================

    def export_memory_snapshot(self) -> Dict[str, Any]:
        """Export complete memory state for debugging/analysis"""
        return {
            "metadata": {
                "question": self.question,
                "created_at": self.created_at,
                "last_updated": self.last_updated,
                "memory_version": "3-layer-v1.0",
            },
            "strategic_layer": {
                "status": self.strategic["status"],
                "plan": [step.to_dict() for step in self.strategic["current_plan"]],
                "rationale": self.strategic["plan_rationale"],
                "strategy": self.strategic["strategy"],
                "replan_history": self.strategic["replan_history"],
                "alternatives": self.strategic["alternative_plans"],
            },
            "execution_layer": {
                "cycles": [
                    cycle.to_dict() for cycle in self.execution["execution_cycles"]
                ],
                "current_state": {
                    "step_id": self.execution["current_step_id"],
                    "prediction": self.execution["current_prediction"],
                    "observation": self.execution["current_observation"],
                    "thought": self.execution["current_thought"],
                },
                "corrections": self.execution["correction_attempts"],
                "metrics": self.execution["performance_metrics"],
            },
            "knowledge_layer": {
                "exploration_graph": self.knowledge["exploration_graph"],
                "entities": self.knowledge["discovered_entities"],
                "paths": [
                    path.to_dict() for path in self.knowledge["exploration_paths"]
                ],
                "findings": self.knowledge["key_findings"],
                "relevance_scores": self.knowledge["entity_relevance_scores"],
            },
            "statistics": self.statistics,
        }

    def save_to_file(self, filepath: str):
        """Save memory snapshot to file"""
        snapshot = self.export_memory_snapshot()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        print(f"💾 Memory snapshot saved to {filepath}")

    def print_memory_status(self):
        """Print current memory status for debugging"""
        summary = self.get_memory_summary()

        print("\n" + "=" * 60)
        print("📚 PPOGA MEMORY STATUS")
        print("=" * 60)
        print(f"Question: {self.question}")
        print(f"Status: {summary['status']}")
        print(
            f"Progress: Step {summary['strategic_summary']['current_step']}/{summary['strategic_summary']['total_steps']}"
        )
        print(
            f"Entities Discovered: {summary['knowledge_summary']['entities_discovered']}"
        )
        print(
            f"Execution Cycles: {summary['execution_summary']['total_cycles']} (Success: {summary['execution_summary']['success_rate']:.1%})"
        )
        print(f"LLM Calls: {summary['statistics']['llm_calls']}")
        print(f"KG Queries: {summary['statistics']['kg_queries']}")
        if summary["strategic_summary"]["replan_count"] > 0:
            print(f"Replans: {summary['strategic_summary']['replan_count']}")
        print("=" * 60)
