import json
import time
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

IMPORTANT: Return ONLY valid JSON without markdown formatting. Do not wrap your response in ```json``` code blocks.
"""

        max_retries = 2
        for attempt in range(max_retries + 1):
            response, token_info = self._call_llm(prompt, self.temperature_reasoning)

            # Clean response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            try:
                plan_data = json.loads(cleaned_response)

                # Validate plan structure
                if "plan" not in plan_data or not isinstance(plan_data["plan"], list):
                    raise ValueError("Invalid plan structure")

                # Ensure each step has required fields
                for i, step in enumerate(plan_data["plan"]):
                    if "step_id" not in step:
                        step["step_id"] = i + 1
                    if "description" not in step:
                        raise ValueError(f"Step {i+1} missing description")

                print(
                    f"✅ Plan decomposition successful: {len(plan_data['plan'])} steps"
                )
                for step in plan_data["plan"]:
                    print(f"   Step {step['step_id']}: {step['description']}")

                return {
                    "success": True,
                    "plan": plan_data["plan"],
                    "rationale": plan_data.get("rationale", ""),
                    "token_usage": token_info,
                }

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    print(f"❌ Plan parsing error (attempt {attempt + 1}): {e}")
                    print(f"Raw response: {response}")
                    print(f"🔄 Retrying with error feedback...")

                    # Add error feedback to prompt for retry
                    prompt = f"""
The previous response failed to parse. Error: {str(e)}
Raw response that failed: {response}

Please fix the JSON format and try again.

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

CRITICAL: Return ONLY valid JSON without any markdown formatting, code blocks, or extra text.
"""
                    continue
                else:
                    print(
                        f"❌ Plan parsing failed after {max_retries + 1} attempts: {e}"
                    )
                    print(f"Final raw response: {response}")
                    break

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

    def observe(
        self,
        raw_observation: str,
        plan_step: Dict[str, Any],
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Process and structure the observation from action execution

        This implements the Observation component of PPoGA methodology
        """
        print(f"👁️ Predictive Planner: Processing observation...")

        prompt = f"""
You are processing the results of a knowledge graph exploration step.

Raw Observation Results: {raw_observation}
Current Plan Step: {plan_step['description']}
Step Objective: {plan_step.get('objective', 'Not specified')}
Memory Context: {memory_context}

Your task is to structure and summarize the observation in a clear format:

1. Extract key entities found
2. Extract key relations discovered  
3. Identify direct answers to the question (if any)
4. Summarize what was learned

Output your structured observation in JSON format:
{{
    "key_entities": ["list of important entities found"],
    "key_relations": ["list of important relations discovered"],
    "direct_answers": ["any direct answers found for the main question"],
    "summary": "brief summary of what was observed",
    "information_quality": "high/medium/low",
    "relevant_to_question": "high/medium/low"
}}
"""

        response, token_info = self._call_llm(prompt, self.temperature_reasoning)

        try:
            observation_data = json.loads(response)

            return {
                "success": True,
                "structured_observation": observation_data,
                "token_usage": token_info,
            }

        except json.JSONDecodeError as e:
            print(f"❌ Observation parsing error: {e}")

            # Fallback observation
            fallback_observation = {
                "key_entities": ["Various entities found"],
                "key_relations": ["Multiple relations explored"],
                "direct_answers": [],
                "summary": (
                    raw_observation[:200] + "..."
                    if len(raw_observation) > 200
                    else raw_observation
                ),
                "information_quality": "medium",
                "relevant_to_question": "medium",
            }

            return {
                "success": False,
                "structured_observation": fallback_observation,
                "token_usage": token_info,
                "error": str(e),
            }

    def think(
        self,
        prediction: Dict[str, Any],
        structured_observation: Dict[str, Any],
        plan_step: Dict[str, Any],
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Analyze and compare prediction with actual observation

        This implements the Think component of PPoGA methodology
        """
        print(f"🤔 Predictive Planner: Thinking about results...")

        prompt = f"""
You are analyzing the results of a knowledge graph exploration step.

QUESTION TO ANSWER: {memory_context.split('Question: ')[1].split('\\n')[0] if 'Question: ' in memory_context else 'Unknown question'}

Original Prediction: {json.dumps(prediction, indent=2)}
Structured Observation: {json.dumps(structured_observation, indent=2)}
Plan Step: {plan_step['description']}
Memory Context: {memory_context}

Analyze the relationship between prediction and observation:

1. How accurate was the prediction compared to what actually happened?
2. What unexpected information was discovered?
3. What important information is missing?
4. How well does this align with our plan step objective?

Output your analysis in JSON format:
{{
    "prediction_accuracy": "high/medium/low",
    "accuracy_details": "explanation of prediction accuracy",
    "unexpected_findings": ["list of unexpected discoveries"],
    "missing_information": ["what we still need to find"],
    "step_alignment": "how well this aligns with the step objective",
    "key_insights": ["important insights gained from this analysis"]
}}
"""

        response, token_info = self._call_llm(prompt, self.temperature_reasoning)

        try:
            thinking_data = json.loads(response)

            return {
                "success": True,
                "thinking_analysis": thinking_data,
                "token_usage": token_info,
            }

        except json.JSONDecodeError as e:
            print(f"❌ Thinking parsing error: {e}")

            # Fallback thinking
            fallback_thinking = {
                "prediction_accuracy": "medium",
                "accuracy_details": "Unable to fully analyze prediction accuracy",
                "unexpected_findings": ["Various unexpected results"],
                "missing_information": ["Additional information may be needed"],
                "step_alignment": "Partial alignment with step objective",
                "key_insights": ["Some insights gained from exploration"],
            }

            return {
                "success": False,
                "thinking_analysis": fallback_thinking,
                "token_usage": token_info,
                "error": str(e),
            }

    def evaluate(
        self,
        thinking_analysis: Dict[str, Any],
        structured_observation: Dict[str, Any],
        plan_step: Dict[str, Any],
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Evaluate progress and decide next action

        This implements the Evaluation component of PPoGA methodology
        """
        print(f"⚖️ Predictive Planner: Evaluating progress and deciding next action...")

        prompt = f"""
You are evaluating progress and deciding the next action in knowledge graph exploration.

QUESTION TO ANSWER: {memory_context.split('Question: ')[1].split('\\n')[0] if 'Question: ' in memory_context else 'Unknown question'}

Thinking Analysis: {json.dumps(thinking_analysis, indent=2)}
Structured Observation: {json.dumps(structured_observation, indent=2)}
Plan Step: {plan_step['description']}
Memory Context: {memory_context}

CRITICAL: Check if the observation contains DIRECT ANSWERS to the main question.

For example:
- If asking "Who directed X?", look for "film.director.film: [Director Name]" in direct_answers
- If asking "When was X born?", look for "people.person.date_of_birth: [Date]" in direct_answers
- If asking "What is the capital of X?", look for "location.country.capital: [City]" in direct_answers

Based on all the analysis, decide the next action:

Next action options:
- PROCEED: Continue to next step in the plan (use this if we found the answer or made significant progress)
- CORRECT_PATH: Retry current step with different approach (only if no relevant information was found)
- REPLAN: Current plan isn't working, need new strategy
- FINISH: Sufficient information gathered to answer the question

Output your evaluation in JSON format:
{{
    "direct_answer_found": "yes/no",
    "answer_content": "the specific answer if found, or 'none' if not found",
    "step_success": "success/partial_success/failure",
    "progress_assessment": "assessment of overall progress toward goal",
    "next_action": "PROCEED/CORRECT_PATH/REPLAN/FINISH",
    "reasoning": "detailed explanation for the decision",
    "confidence": "high/medium/low",
    "information_completeness": "complete/partial/insufficient"
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
                "direct_answer_found": "unknown",
                "answer_content": "none",
                "step_success": "partial_success",
                "progress_assessment": "Making some progress",
                "next_action": "PROCEED",
                "reasoning": "Fallback decision due to parsing error",
                "confidence": "low",
                "information_completeness": "partial",
            }

            return {
                "success": False,
                "evaluation": fallback_evaluation,
                "next_action": "PROCEED",
                "error": str(e),
                "token_usage": token_info,
            }

    def observe_and_think(
        self,
        prediction: Dict[str, Any],
        observation: str,
        plan_step: Dict[str, Any],
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        4단계: 관찰 및 사고 (Observation and Thought)

        Executor로부터 보고받은 Observation을 Prediction과 비교 분석하여
        예측 오류(Prediction Error)를 계산하고 계획의 진행 상황을 평가함
        """
        print(
            f"🧠 Predictive Planner: Analyzing observation and generating thoughts..."
        )

        prompt = f"""
You are analyzing the results of a knowledge graph exploration step according to PPoGA methodology.

QUESTION TO ANSWER: {memory_context.split('Question: ')[1].split('\\n')[0] if 'Question: ' in memory_context else 'Unknown question'}

Original Prediction: {json.dumps(prediction, indent=2)}
Actual Observation: {observation}
Plan Step: {plan_step['description']}
Memory Context: {memory_context}

Your task is to perform OBSERVATION AND THOUGHT analysis (Step 4 of PPoGA):

1. Compare the prediction with the actual observation
2. Calculate the "prediction error" - how accurate was your prediction?
3. Analyze what specific information was gained
4. Assess the progress toward the overall goal
5. Generate thoughts about the current situation

IMPORTANT: This is NOT the evaluation step. Do not decide next actions yet. 
Focus only on understanding and thinking about what happened.

Output your analysis in JSON format:
{{
    "prediction_accuracy": "high/medium/low",
    "prediction_error_analysis": "detailed analysis of what was predicted vs what actually happened",
    "information_gained": "specific information discovered in this step", 
    "direct_answer_found": "yes/no",
    "answer_content": "the specific answer if found, or 'none' if not found",
    "progress_assessment": "assessment of progress toward the overall goal",
    "current_thoughts": "your thoughts and insights about the current situation",
    "confidence_in_findings": "high/medium/low"
}}
"""

        response, token_info = self._call_llm(prompt, self.temperature_reasoning)

        try:
            thought_data = json.loads(response)

            return {
                "success": True,
                "thought_analysis": thought_data,
                "token_usage": token_info,
            }

        except json.JSONDecodeError as e:
            print(f"❌ Thought analysis parsing error: {e}")

            # Fallback thought analysis
            fallback_analysis = {
                "prediction_accuracy": "unknown",
                "prediction_error_analysis": "Could not analyze due to parsing error",
                "information_gained": (
                    observation[:100] + "..." if len(observation) > 100 else observation
                ),
                "direct_answer_found": "unknown",
                "answer_content": "none",
                "progress_assessment": "Unable to assess due to parsing error",
                "current_thoughts": "Analysis failed, using fallback assessment",
                "confidence_in_findings": "low",
            }

            return {
                "success": False,
                "thought_analysis": fallback_analysis,
                "error": str(e),
                "token_usage": token_info,
            }

    def evaluate_and_correct(
        self,
        thought_analysis: Dict[str, Any],
        plan_step: Dict[str, Any],
        memory_context: str,
        current_plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        6단계: 평가 및 2단계 자기 교정 (Evaluation & Two-Level Self-Correction)

        Thought 분석을 바탕으로 최종 행동을 결정하고 2단계 자기 교정을 수행함
        """
        print(
            f"⚖️ Predictive Planner: Evaluating results and determining correction strategy..."
        )

        prompt = f"""
You are making the final decision for the current execution cycle according to PPoGA methodology.

QUESTION TO ANSWER: {memory_context.split('Question: ')[1].split('\\n')[0] if 'Question: ' in memory_context else 'Unknown question'}

Thought Analysis: {json.dumps(thought_analysis, indent=2)}
Current Plan Step: {plan_step['description']}
Overall Plan: {json.dumps(current_plan, indent=2)}
Memory Context: {memory_context}

Your task is to perform EVALUATION AND TWO-LEVEL SELF-CORRECTION (Step 6 of PPoGA):

Based on the thought analysis, decide the next action using PPoGA's two-level correction:

DECISION OPTIONS:
- PROCEED: Continue to next step in the plan (normal progression)
- TACTICAL_CORRECTION: Current plan is valid but execution path needs adjustment (Level 1 - analytical solving)
- STRATEGIC_CORRECTION: Current plan itself is flawed and needs complete replanning (Level 2 - problem restructuring)  
- FINISH: Sufficient information gathered to answer the question

CORRECTION LEVELS:
- Level 1 (Tactical): Retry same plan step with different approach/relations
- Level 2 (Strategic): Abandon current plan and create entirely new strategy

Output your evaluation in JSON format:
{{
    "step_success": "success/partial_success/failure",
    "overall_progress": "assessment of progress toward final goal",
    "correction_needed": "none/tactical/strategic",
    "next_action": "PROCEED/TACTICAL_CORRECTION/STRATEGIC_CORRECTION/FINISH",
    "reasoning": "detailed explanation for the decision",
    "confidence": "high/medium/low",
    "correction_strategy": "if correction needed, describe the strategy"
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
                "step_success": "partial_success",
                "overall_progress": "Making some progress",
                "correction_needed": "none",
                "next_action": "PROCEED",
                "reasoning": "Fallback decision due to parsing error",
                "confidence": "low",
                "correction_strategy": "Continue with current approach",
            }

            return {
                "success": False,
                "evaluation": fallback_evaluation,
                "next_action": "PROCEED",
                "error": str(e),
                "token_usage": token_info,
            }

    def think_and_evaluate(
        self,
        prediction: Dict[str, Any],
        observation: str,
        plan_step: Dict[str, Any],
        memory_context: str,
    ) -> Dict[str, Any]:
        """
        Legacy method that combines think and evaluate - now implements proper PPoGA sequence

        This method now properly implements: Observation → Think → Evaluation
        """
        print(
            f"� Predictive Planner: Running PPoGA sequence (Observe → Think → Evaluate)..."
        )

        # Step 1: Observe - Process raw observation
        observe_result = self.observe(observation, plan_step, memory_context)
        structured_observation = observe_result.get("structured_observation", {})

        # Step 2: Think - Analyze prediction vs observation
        think_result = self.think(
            prediction, structured_observation, plan_step, memory_context
        )
        thinking_analysis = think_result.get("thinking_analysis", {})

        # Step 3: Evaluate - Decide next action
        evaluate_result = self.evaluate(
            thinking_analysis, structured_observation, plan_step, memory_context
        )

        # Combine all results for backward compatibility
        combined_result = {
            "success": evaluate_result.get("success", True),
            "observation_step": observe_result,
            "thinking_step": think_result,
            "evaluation_step": evaluate_result,
            "evaluation": evaluate_result.get("evaluation", {}),
            "next_action": evaluate_result.get("next_action", "PROCEED"),
            "token_usage": {
                "observe": observe_result.get("token_usage", {}),
                "think": think_result.get("token_usage", {}),
                "evaluate": evaluate_result.get("token_usage", {}),
                "total": (
                    observe_result.get("token_usage", {}).get("total", 0)
                    + think_result.get("token_usage", {}).get("total", 0)
                    + evaluate_result.get("token_usage", {}).get("total", 0)
                ),
            },
        }

        return combined_result

        prompt = f"""
You are evaluating the results of a knowledge graph exploration step.

QUESTION TO ANSWER: {memory_context.split('Question: ')[1].split('\\n')[0] if 'Question: ' in memory_context else 'Unknown question'}

Original Prediction: {json.dumps(prediction, indent=2)}
Actual Observation: {observation}
Plan Step: {plan_step['description']}
Memory Context: {memory_context}

CRITICAL: Carefully examine the observation for direct answers to the question. Look for specific relation-value pairs that directly answer what was asked.

For example:
- If asking "Who directed X?", look for "film.director.film: [Director Name]"
- If asking "When was X born?", look for "people.person.date_of_birth: [Date]"
- If asking "What is the capital of X?", look for "location.country.capital: [City]"

Compare the prediction with the actual observation and provide your analysis:

1. Does the observation contain a DIRECT ANSWER to the main question?
2. How accurate was the prediction?
3. What specific information was gained?
4. Are we making progress toward answering the question?
5. What should be the next action?

Next action options:
- PROCEED: Continue to next step in the plan (use this if we found the answer or made significant progress)
- CORRECT_PATH: Retry current step with different approach (only if no relevant information was found)
- REPLAN: Current plan isn't working, need new strategy
- FINISH: Sufficient information gathered to answer the question

Output your evaluation in JSON format:
{{
    "direct_answer_found": "yes/no",
    "answer_content": "the specific answer if found, or 'none' if not found",
    "prediction_accuracy": "high/medium/low",
    "step_success": "success/partial_success/failure",
    "information_gained": "description of what was learned",
    "progress_assessment": "assessment of progress toward goal",
    "next_action": "PROCEED/CORRECT_PATH/REPLAN/FINISH",
    "reasoning": "explanation for the decision, especially if answer was found",
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
            opeani_api_keys=self.api_key,
            engine=self.model,
            print_in=False,
            print_out=False,
        )

        self.stats["total_llm_calls"] += 1
        return response, token_info

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics"""
        return self.stats.copy()
