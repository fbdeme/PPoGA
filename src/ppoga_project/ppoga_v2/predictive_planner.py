import json
import time
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from PoG.utils import run_llm


class PredictivePlanner:
    """
    Predictive Planner that combines PoG planning with PreAct prediction capabilities

    This planner extends PoG's strategic planning with:
    1. Outcome prediction before action (PreAct)
    2. Dynamic replanning capabilities (PLAN-AND-ACT)
    3. Enhanced self-correction mechanisms
    """

    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.api_key = llm_config.get("api_key", "")
        self.model = llm_config.get("model", "gpt-3.5-turbo")
        self.temperature_exploration = llm_config.get("temperature_exploration", 0.3)
        self.temperature_reasoning = llm_config.get("temperature_reasoning", 0.3)
        self.max_length = llm_config.get("max_length", 4096)

        # Statistics
        self.stats = {
            "total_llm_calls": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "replans_triggered": 0,
        }

    def decompose_plan(
        self, question: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Decompose question into strategic plan

        Extends PoG's subobjective decomposition with strategic planning
        """
        print("🧐 Predictive Planner: Decomposing question into strategic plan...")

        prompt = f"""
You are a strategic planner for knowledge graph exploration. Break down the following complex question into a detailed strategic plan.

Question: {question}

Your task is to create a step-by-step plan where each step has:
1. A clear objective
2. Expected outcome
3. Specific actions to take

Output your plan in JSON format with the following structure:
{{
    "plan": [
        {{
            "step_id": 1,
            "description": "Brief description of what to do",
            "objective": "Specific goal for this step",
            "expected_outcome": "What information we expect to gain"
        }},
        ...
    ],
    "rationale": "Why this plan will effectively answer the question"
}}

Focus on creating 3-5 strategic steps that build upon each other.
"""

        response, token_info = self._call_llm(prompt, self.temperature_reasoning)

        try:
            plan_data = json.loads(response)

            # Validate plan structure
            if "plan" not in plan_data or not isinstance(plan_data["plan"], list):
                raise ValueError("Invalid plan structure")

            # Ensure each step has required fields
            for i, step in enumerate(plan_data["plan"]):
                if "step_id" not in step:
                    step["step_id"] = i + 1
                if "description" not in step:
                    raise ValueError(f"Step {i+1} missing description")

            print(f"✅ Plan decomposition successful: {len(plan_data['plan'])} steps")
            for step in plan_data["plan"]:
                print(f"   Step {step['step_id']}: {step['description']}")

            return {
                "success": True,
                "plan": plan_data["plan"],
                "rationale": plan_data.get("rationale", ""),
                "token_usage": token_info,
            }

        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Plan parsing error: {e}")
            print(f"Raw response: {response}")

            # Fallback: create a simple plan
            fallback_plan = [
                {
                    "step_id": 1,
                    "description": "Identify key entities in the question",
                    "objective": "Find main entities to explore",
                    "expected_outcome": "List of relevant entities",
                },
                {
                    "step_id": 2,
                    "description": "Explore relationships of key entities",
                    "objective": "Discover connected information",
                    "expected_outcome": "Related entities and properties",
                },
                {
                    "step_id": 3,
                    "description": "Synthesize information to answer question",
                    "objective": "Combine findings into final answer",
                    "expected_outcome": "Complete answer to the question",
                },
            ]

            return {
                "success": False,
                "plan": fallback_plan,
                "rationale": "Fallback plan due to parsing error",
                "token_usage": token_info,
                "error": str(e),
            }

    def predict_step_outcome(
        self, plan_step: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict the outcome of executing a plan step (PreAct inspiration)

        This implements the prediction component of PreAct methodology
        """
        print(
            f"🔮 Predictive Planner: Predicting outcome for step {plan_step['step_id']}..."
        )

        prompt = f"""
You are predicting the outcome of a knowledge graph exploration step.

Current Step: {plan_step['description']}
Objective: {plan_step.get('objective', 'Not specified')}
Context: {context.get('current_context', 'Initial exploration')}

Based on the step description and context, predict:
1. What entities or information will likely be discovered
2. What challenges might arise
3. How confident you are in finding relevant information
4. What alternative approaches might be needed

Output your prediction in JSON format:
{{
    "predicted_entities": ["list of entities you expect to find"],
    "predicted_challenges": ["potential issues or obstacles"],
    "confidence_level": "high/medium/low",
    "alternative_approaches": ["backup plans if primary approach fails"],
    "expected_information_type": "description of what kind of info to expect"
}}
"""

        response, token_info = self._call_llm(prompt, self.temperature_exploration)

        try:
            prediction_data = json.loads(response)
            self.stats["successful_predictions"] += 1

            return {
                "success": True,
                "prediction": prediction_data,
                "token_usage": token_info,
            }

        except json.JSONDecodeError as e:
            print(f"❌ Prediction parsing error: {e}")
            self.stats["failed_predictions"] += 1

            # Fallback prediction
            fallback_prediction = {
                "predicted_entities": ["Unknown entities related to the question"],
                "predicted_challenges": ["Information might not be directly available"],
                "confidence_level": "medium",
                "alternative_approaches": ["Try different relations or entities"],
                "expected_information_type": "Entities and relations relevant to the question",
            }

            return {
                "success": False,
                "prediction": fallback_prediction,
                "token_usage": token_info,
                "error": str(e),
            }

    def think_and_evaluate(
        self,
        prediction: Dict[str, Any],
        observation: str,
        plan_step: Dict[str, Any],
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Compare prediction with actual observation and decide next action

        This implements the thought component that compares predicted vs actual outcomes
        """
        print(
            f"🤔 Predictive Planner: Analyzing results and determining next action..."
        )

        prompt = f"""
You are evaluating the results of a knowledge graph exploration step.

Original Prediction: {json.dumps(prediction, indent=2)}
Actual Observation: {observation}
Plan Step: {plan_step['description']}
Memory Context: {memory_context}

Compare the prediction with the actual observation and provide your analysis:

1. How accurate was the prediction?
2. What information was gained?
3. Are we making progress toward answering the question?
4. What should be the next action?

Next action options:
- PROCEED: Continue to next step in the plan
- CORRECT_PATH: Retry current step with different approach
- REPLAN: Current plan isn't working, need new strategy
- FINISH: Sufficient information gathered

Output your evaluation in JSON format:
{{
    "prediction_accuracy": "high/medium/low",
    "step_success": "success/partial_success/failure",
    "information_gained": "description of what was learned",
    "progress_assessment": "assessment of progress toward goal",
    "next_action": "PROCEED/CORRECT_PATH/REPLAN/FINISH",
    "reasoning": "explanation for the decision",
    "confidence": "high/medium/low"
}}
"""

        response, token_info = self._call_llm(prompt, self.temperature_reasoning)

        try:
            evaluation_data = json.loads(response)

            return {
                "success": True,
                "evaluation": evaluation_data,
                "next_action": evaluation_data.get("next_action", "PROCEED"),
                "token_usage": token_info,
            }

        except json.JSONDecodeError as e:
            print(f"❌ Evaluation parsing error: {e}")

            # Fallback evaluation
            fallback_evaluation = {
                "prediction_accuracy": "unknown",
                "step_success": "partial_success",
                "information_gained": (
                    observation[:100] + "..." if len(observation) > 100 else observation
                ),
                "progress_assessment": "Making some progress",
                "next_action": "PROCEED",
                "reasoning": "Fallback decision due to parsing error",
                "confidence": "low",
            }

            return {
                "success": False,
                "evaluation": fallback_evaluation,
                "next_action": "PROCEED",
                "error": str(e),
                "token_usage": token_info,
            }

    def replan(
        self,
        question: str,
        failed_plan: List[Dict[str, Any]],
        failure_reason: str,
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Create a new strategic plan based on previous failures

        This implements dynamic replanning from PLAN-AND-ACT methodology
        """
        print(f"🔄 Predictive Planner: Creating new strategic plan...")
        self.stats["replans_triggered"] += 1

        prompt = f"""
The current strategic plan has failed and needs to be replaced with a new approach.

Original Question: {question}
Failed Plan: {json.dumps(failed_plan, indent=2)}
Failure Reason: {failure_reason}
Memory Context: {memory_context}

Create a completely new strategic plan that:
1. Learns from the previous failure
2. Takes a different approach
3. Is more likely to succeed

Output your new plan in JSON format:
{{
    "new_plan": [
        {{
            "step_id": 1,
            "description": "Brief description of what to do",
            "objective": "Specific goal for this step",
            "expected_outcome": "What information we expect to gain"
        }},
        ...
    ],
    "strategy_change": "explanation of how this plan differs from the failed one",
    "confidence": "high/medium/low"
}}
"""

        response, token_info = self._call_llm(prompt, self.temperature_reasoning)

        try:
            replan_data = json.loads(response)

            return {
                "success": True,
                "new_plan": replan_data.get("new_plan", []),
                "strategy_change": replan_data.get("strategy_change", "New approach"),
                "confidence": replan_data.get("confidence", "medium"),
                "token_usage": token_info,
            }

        except json.JSONDecodeError as e:
            print(f"❌ Replan parsing error: {e}")

            # Fallback replan
            fallback_plan = [
                {
                    "step_id": 1,
                    "description": "Try alternative entity identification approach",
                    "objective": "Find different relevant entities",
                    "expected_outcome": "New entities to explore",
                },
                {
                    "step_id": 2,
                    "description": "Explore different types of relationships",
                    "objective": "Discover alternative information paths",
                    "expected_outcome": "Different connected information",
                },
            ]

            return {
                "success": False,
                "new_plan": fallback_plan,
                "strategy_change": "Fallback alternative approach",
                "confidence": "low",
                "token_usage": token_info,
                "error": str(e),
            }

    def generate_final_answer(
        self, question: str, memory_context: str, reasoning_chains: List[Any]
    ) -> Dict[str, Any]:
        """Generate final answer based on all collected information"""
        print(f"📝 Predictive Planner: Generating final answer...")

        prompt = f"""
Based on all the exploration and reasoning, provide a final answer to the question.

Question: {question}
Memory Context: {memory_context}
Reasoning Chains: {json.dumps(reasoning_chains, indent=2)}

Provide your final answer in JSON format:
{{
    "final_answer": "the answer to the question",
    "confidence": "high/medium/low",
    "supporting_evidence": ["key evidence supporting this answer"],
    "reasoning_chain": "step-by-step reasoning that led to this answer",
    "limitations": "any limitations or uncertainties in the answer"
}}
"""

        response, token_info = self._call_llm(prompt, self.temperature_reasoning)

        try:
            answer_data = json.loads(response)

            return {
                "success": True,
                "answer_data": answer_data,
                "token_usage": token_info,
            }

        except json.JSONDecodeError as e:
            print(f"❌ Answer parsing error: {e}")

            fallback_answer = {
                "final_answer": "Unable to determine answer from available information",
                "confidence": "low",
                "supporting_evidence": ["Insufficient or unclear information"],
                "reasoning_chain": "Could not establish clear reasoning chain",
                "limitations": "Limited by information retrieval and parsing issues",
            }

            return {
                "success": False,
                "answer_data": fallback_answer,
                "token_usage": token_info,
                "error": str(e),
            }

    def _call_llm(self, prompt: str, temperature: float) -> Tuple[str, Dict[str, int]]:
        """Internal method to call LLM"""
        response, token_info = run_llm(
            prompt=prompt,
            temperature=temperature,
            max_tokens=self.max_length,
            openai_api_keys=self.api_key,
            engine=self.model,
            print_in=False,
            print_out=False,
        )

        self.stats["total_llm_calls"] += 1
        return response, token_info

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return self.stats.copy()
