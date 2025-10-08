"""
PPoGA Predictive Planner
Combines PoG planning with PreAct prediction capabilities
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from .azure_llm import (
    call_azure_openai,
    extract_json_from_response,
    extract_list_from_response,
    global_token_tracker,
)


@dataclass
class PlanStep:
    """Individual plan step with prediction capabilities"""

    step_id: int
    description: str
    objective: str
    expected_entities: List[str]
    predicted_outcome: Optional[Dict[str, Any]] = None
    confidence: float = 0.5


@dataclass
class ExecutionCycle:
    """Single prediction-action-observation-thought cycle"""

    step_id: int
    prediction: Dict[str, Any]
    action: str
    observation: str
    thought: Dict[str, Any]
    timestamp: float
    success: bool = False


class PredictivePlanner:
    """
    Enhanced Planner that combines PoG planning with PreAct prediction
    """

    def __init__(self, azure_config: Dict[str, Any]):
        self.azure_config = azure_config
        self.token_tracker = global_token_tracker

    def decompose_plan_with_prediction(
        self, question: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Decompose question into plan with prediction for each step
        """
        print(f"🧠 Predictive Planner: Decomposing question with prediction...")

        # Enhanced prompt that includes prediction
        prompt = self._create_plan_decomposition_prompt(question, context)
        response, token_usage = call_azure_openai(
            prompt, self.azure_config, temperature=0.3
        )
        self.token_tracker.add_usage(token_usage)

        try:
            plan_data = extract_json_from_response(response)

            if "fallback" in plan_data:
                # Use fallback plan if JSON parsing failed
                return self._create_fallback_plan(question)

            # Validate and enhance plan with predictions
            enhanced_plan = []
            for i, step in enumerate(plan_data.get("plan", [])):
                enhanced_step = PlanStep(
                    step_id=i + 1,
                    description=step.get("description", f"Step {i+1}"),
                    objective=step.get("objective", ""),
                    expected_entities=step.get("expected_entities", []),
                    predicted_outcome=step.get("predicted_outcome", {}),
                    confidence=step.get("confidence", 0.5),
                )
                enhanced_plan.append(enhanced_step.__dict__)

            print(
                f"✅ Plan decomposition successful: {len(enhanced_plan)} steps with predictions"
            )
            for step in enhanced_plan:
                print(f"   Step {step['step_id']}: {step['description']}")
                print(
                    f"     → Predicted: {step['predicted_outcome'].get('summary', 'No prediction')}"
                )

            return {
                "success": True,
                "plan": enhanced_plan,
                "rationale": plan_data.get("rationale", ""),
                "strategy": plan_data.get("strategy", ""),
                "token_usage": token_usage,
            }

        except Exception as e:
            print(f"❌ Plan decomposition error: {e}")
            return self._create_fallback_plan(question)

    def predict_step_outcome(
        self, plan_step: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict outcome of a plan step before execution (PreAct inspiration)
        """
        print(f"🔮 Predicting outcome for Step {plan_step.get('step_id', 'N/A')}")

        prompt = self._create_prediction_prompt(plan_step, context)
        response, token_usage = call_azure_openai(
            prompt, self.azure_config, temperature=0.2
        )
        self.token_tracker.add_usage(token_usage)

        try:
            prediction_data = extract_json_from_response(response)

            if "fallback" in prediction_data:
                # Fallback prediction
                prediction_data = {
                    "expected_entities": ["unknown_entity"],
                    "expected_relations": ["unknown_relation"],
                    "confidence": 0.3,
                    "reasoning": "Fallback prediction due to parsing error",
                    "success_probability": 0.5,
                }

            print(
                f"   Predicted entities: {prediction_data.get('expected_entities', [])}"
            )
            print(f"   Confidence: {prediction_data.get('confidence', 0)}")

            return {
                "success": True,
                "prediction": prediction_data,
                "token_usage": token_usage,
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                "success": False,
                "prediction": {"confidence": 0.1, "reasoning": f"Error: {e}"},
                "token_usage": token_usage,
            }

    def think_and_evaluate(
        self,
        prediction: Dict[str, Any],
        observation: str,
        plan_step: Dict[str, Any],
        memory_summary: Dict[str, Any],
        question: str,
    ) -> Dict[str, Any]:
        """
        Compare prediction with actual observation and decide next action
        """
        print(f"🤔 Evaluating Step {plan_step.get('step_id', 'N/A')} results...")

        prompt = self._create_evaluation_prompt(
            prediction, observation, plan_step, memory_summary, question
        )
        response, token_usage = call_azure_openai(
            prompt, self.azure_config, temperature=0.3
        )
        self.token_tracker.add_usage(token_usage)

        try:
            evaluation_data = extract_json_from_response(response)

            if "fallback" in evaluation_data:
                # Fallback evaluation
                evaluation_data = {
                    "prediction_accuracy": "unknown",
                    "step_success": "partial_success",
                    "information_gained": (
                        observation[:100] + "..."
                        if len(observation) > 100
                        else observation
                    ),
                    "answer_feasibility": "partial",
                    "next_action": "PROCEED",
                    "reasoning": "Fallback decision due to parsing error",
                    "confidence": 0.3,
                }

            next_action = evaluation_data.get("next_action", "PROCEED")
            print(f"   Decision: {next_action}")
            print(
                f"   Reasoning: {evaluation_data.get('reasoning', 'No reasoning provided')}"
            )

            return {
                "success": True,
                "evaluation": evaluation_data,
                "next_action": next_action,
                "token_usage": token_usage,
            }

        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            return {
                "success": False,
                "evaluation": {"confidence": 0.1},
                "next_action": "PROCEED",
                "error": str(e),
                "token_usage": token_usage,
            }

    def replan(
        self,
        question: str,
        failed_plan: List[Dict[str, Any]],
        failure_reason: str,
        memory_summary: Dict[str, Any],
        previous_attempts: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create new strategic plan based on previous failures (PLAN-AND-ACT inspiration)
        """
        print(f"🔄 Strategic replanning due to: {failure_reason}")

        prompt = self._create_replan_prompt(
            question, failed_plan, failure_reason, memory_summary, previous_attempts
        )
        response, token_usage = call_azure_openai(
            prompt, self.azure_config, temperature=0.5
        )
        self.token_tracker.add_usage(token_usage)

        try:
            replan_data = extract_json_from_response(response)

            if "fallback" in replan_data:
                # Fallback to simplified plan
                return self._create_fallback_plan(question, is_replan=True)

            new_plan = replan_data.get("new_plan", [])
            strategy_change = replan_data.get("strategy_change", "No strategy change")

            print(f"✅ Replanning successful: {len(new_plan)} new steps")
            print(f"   Strategy change: {strategy_change}")

            return {
                "success": True,
                "new_plan": new_plan,
                "strategy_change": strategy_change,
                "rationale": replan_data.get("rationale", ""),
                "token_usage": token_usage,
            }

        except Exception as e:
            print(f"❌ Replanning error: {e}")
            return self._create_fallback_plan(question, is_replan=True)

    def generate_final_answer(
        self,
        question: str,
        knowledge_summary: Dict[str, Any],
        reasoning_paths: List[Dict[str, Any]],
        exploration_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate final answer based on collected knowledge
        """
        print(f"📝 Generating final answer...")

        prompt = self._create_final_answer_prompt(
            question, knowledge_summary, reasoning_paths, exploration_history
        )
        response, token_usage = call_azure_openai(
            prompt, self.azure_config, temperature=0.2
        )
        self.token_tracker.add_usage(token_usage)

        try:
            answer_data = extract_json_from_response(response)

            if "fallback" in answer_data:
                # Fallback answer
                answer_data = {
                    "final_answer": "Unable to determine answer due to processing error",
                    "confidence": "low",
                    "reasoning_chain": "Error in final answer generation",
                    "supporting_evidence": [],
                    "limitations": "Technical error in answer synthesis",
                }

            print(f"✅ Final answer generated")
            print(f"   Answer: {answer_data.get('final_answer', 'No answer')}")
            print(f"   Confidence: {answer_data.get('confidence', 'unknown')}")

            return {
                "success": True,
                "answer_data": answer_data,
                "token_usage": token_usage,
            }

        except Exception as e:
            print(f"❌ Final answer error: {e}")
            return {
                "success": False,
                "answer_data": {
                    "final_answer": f"Error generating answer: {e}",
                    "confidence": "very_low",
                },
                "token_usage": token_usage,
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return {
            "token_usage": self.token_tracker.get_summary(),
            "last_updated": time.time(),
        }

    # Helper methods for prompt creation

    def _create_plan_decomposition_prompt(
        self, question: str, context: Dict[str, Any] = None
    ) -> str:
        """Create enhanced prompt for plan decomposition with prediction"""
        return f"""You are an expert knowledge graph query planner. Decompose the following question into a step-by-step plan.
For each step, predict what entities and relations you expect to find, and estimate confidence.

Question: {question}
Context: {context or "No additional context"}

Create a plan with the following JSON format:
{{
    "plan": [
        {{
            "step_id": 1,
            "description": "Clear description of what to do",
            "objective": "What we hope to achieve",
            "expected_entities": ["entity1", "entity2"],
            "predicted_outcome": {{
                "summary": "What we expect to find",
                "confidence": 0.8,
                "reasoning": "Why we expect this"
            }}
        }}
    ],
    "rationale": "Overall strategy explanation",
    "strategy": "High-level approach"
}}"""

    def _create_prediction_prompt(
        self, plan_step: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Create prompt for step outcome prediction"""
        return f"""Predict the outcome of executing this knowledge graph query step.

Step: {plan_step.get('description', 'Unknown step')}
Objective: {plan_step.get('objective', 'Unknown objective')}
Context: {context}

Predict in JSON format:
{{
    "expected_entities": ["list of entities you expect to find"],
    "expected_relations": ["list of relations you expect to use"],
    "confidence": 0.7,
    "reasoning": "Why you expect these results",
    "success_probability": 0.8,
    "potential_issues": ["possible problems"]
}}"""

    def _create_evaluation_prompt(
        self,
        prediction: Dict[str, Any],
        observation: str,
        plan_step: Dict[str, Any],
        memory_summary: Dict[str, Any],
        question: str,
    ) -> str:
        """Create prompt for prediction vs observation evaluation"""
        return f"""Compare the prediction with actual observation and decide next action.

Original Question: {question}
Step: {plan_step.get('description', 'Unknown step')}
Prediction: {prediction}
Actual Observation: {observation}
Memory Summary: {memory_summary}

Evaluate in JSON format:
{{
    "prediction_accuracy": "high/medium/low",
    "step_success": "success/partial_success/failure",
    "information_gained": "Summary of new information",
    "answer_feasibility": "can_answer/partial/need_more",
    "next_action": "PROCEED/CORRECT_PATH/REPLAN/FINISH",
    "reasoning": "Detailed reasoning for decision",
    "confidence": 0.8
}}

Next actions:
- PROCEED: Continue to next step
- CORRECT_PATH: Retry current step with different approach  
- REPLAN: Current strategy isn't working, need new plan
- FINISH: Have enough information to answer"""

    def _create_replan_prompt(
        self,
        question: str,
        failed_plan: List[Dict[str, Any]],
        failure_reason: str,
        memory_summary: Dict[str, Any],
        previous_attempts: List[Dict[str, Any]] = None,
    ) -> str:
        """Create prompt for strategic replanning"""
        return f"""The current plan has failed. Create a new strategic approach.

Question: {question}
Failed Plan: {failed_plan}
Failure Reason: {failure_reason}
Current Knowledge: {memory_summary}
Previous Attempts: {previous_attempts or "None"}

Create new plan in JSON format:
{{
    "new_plan": [
        {{
            "step_id": 1,
            "description": "New approach description",
            "objective": "What to achieve",
            "expected_entities": ["entities"],
            "rationale": "Why this approach will work better"
        }}
    ],
    "strategy_change": "How this strategy differs from previous attempts",
    "rationale": "Why this new approach should succeed"
}}"""

    def _create_final_answer_prompt(
        self,
        question: str,
        knowledge_summary: Dict[str, Any],
        reasoning_paths: List[Dict[str, Any]],
        exploration_history: List[Dict[str, Any]],
    ) -> str:
        """Create prompt for final answer generation"""
        return f"""Generate the final answer based on all collected knowledge.

Question: {question}
Knowledge Summary: {knowledge_summary}
Reasoning Paths: {reasoning_paths}
Exploration History: {exploration_history}

Generate answer in JSON format:
{{
    "final_answer": "The definitive answer to the question",
    "confidence": "high/medium/low/very_low",
    "reasoning_chain": "Step-by-step reasoning that led to this answer",
    "supporting_evidence": ["key facts that support this answer"],
    "limitations": "Any limitations or uncertainties in this answer"
}}"""

    def _create_fallback_plan(
        self, question: str, is_replan: bool = False
    ) -> Dict[str, Any]:
        """Create a simple fallback plan when main planning fails"""
        prefix = "Replan: " if is_replan else ""

        fallback_plan = [
            {
                "step_id": 1,
                "description": f"{prefix}Identify key entities in the question",
                "objective": "Find main entities to start exploration",
                "expected_entities": ["key_entity"],
                "predicted_outcome": {
                    "summary": "Find starting entities",
                    "confidence": 0.7,
                    "reasoning": "Simple entity identification",
                },
            },
            {
                "step_id": 2,
                "description": f"{prefix}Explore relations of key entities",
                "objective": "Discover relevant connections",
                "expected_entities": ["related_entity"],
                "predicted_outcome": {
                    "summary": "Find connected entities",
                    "confidence": 0.6,
                    "reasoning": "Basic relation exploration",
                },
            },
        ]

        return {
            "success": True,
            "plan": fallback_plan,
            "rationale": f"Fallback plan due to parsing error ({'replanning' if is_replan else 'initial planning'})",
            "strategy": "Simple exploration strategy",
            "token_usage": {"total": 0, "input": 0, "output": 0},
        }
