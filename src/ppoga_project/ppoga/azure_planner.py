"""
Azure OpenAI Enhanced PPoGA Planner Module

This module implements the PPoGA Planner using Azure OpenAI API
with enhanced prompts based on the original papers.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from openai import AzureOpenAI

from .enhanced_prompts import (
    DECOMPOSE_PLAN_PROMPT,
    PREDICT_PROMPT,
    THINK_AND_EVALUATE_PROMPT,
    REPLAN_PROMPT,
    FINAL_ANSWER_PROMPT,
    RELATION_SELECTION_PROMPT
)


class AzurePPoGAPlanner:
    """
    Enhanced PPoGA Planner using Azure OpenAI API
    
    This class implements the strategic planning component of PPoGA,
    handling plan decomposition, prediction, evaluation, and replanning.
    """
    
    def __init__(self, azure_config: Dict[str, Any]):
        """
        Initialize the Azure OpenAI Planner
        
        Args:
            azure_config: Azure OpenAI configuration dictionary
        """
        self.azure_config = azure_config
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=azure_config.get("api_key"),
            api_version=azure_config.get("api_version", "2024-05-01-preview"),
            azure_endpoint=azure_config.get("api_base")
        )
        
        self.deployment_id = azure_config.get("deployment_id", "ktc-nego-0919")
        self.temperature = azure_config.get("temperature", 0.3)
        self.max_tokens = azure_config.get("max_tokens", 2048)
        
        # Statistics tracking
        self.total_llm_calls = 0
        self.total_tokens_used = 0
    
    def _call_azure_openai(self, prompt: str, temperature: Optional[float] = None) -> Tuple[str, Dict[str, int]]:
        """
        Call Azure OpenAI API with error handling and token tracking
        
        Args:
            prompt: The prompt to send to the model
            temperature: Temperature for generation (optional)
            
        Returns:
            Tuple of (response_text, token_usage_info)
        """
        temp = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[
                    {"role": "system", "content": "You are an expert AI assistant specialized in knowledge graph question answering and strategic planning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=self.max_tokens
            )
            
            # Extract response and usage information
            response_text = response.choices[0].message.content
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Update statistics
            self.total_llm_calls += 1
            self.total_tokens_used += usage_info["total_tokens"]
            
            return response_text, usage_info
            
        except Exception as e:
            print(f"❌ Azure OpenAI API Error: {e}")
            return f"Error: {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def decompose_plan(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decompose a complex question into a step-by-step plan
        
        Args:
            question: The complex question to decompose
            context: Additional context information
            
        Returns:
            Dictionary containing the plan and rationale
        """
        print(f"🧐 Azure Planner: Decomposing question into strategic plan...")
        
        prompt = DECOMPOSE_PLAN_PROMPT.format(question=question)
        response, token_info = self._call_azure_openai(prompt)
        
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
                "token_usage": token_info
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Plan parsing error: {e}")
            print(f"Raw response: {response}")
            
            # Fallback: create a simple plan
            fallback_plan = [
                {"step_id": 1, "description": f"Identify key entities in the question: {question}"},
                {"step_id": 2, "description": "Explore relationships and properties of identified entities"},
                {"step_id": 3, "description": "Synthesize information to answer the question"}
            ]
            
            return {
                "success": False,
                "plan": fallback_plan,
                "rationale": "Fallback plan due to parsing error",
                "error": str(e),
                "token_usage": token_info
            }
    
    def predict_step_outcome(self, plan_step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the outcome of executing a specific plan step
        
        Args:
            plan_step: The plan step to predict
            context: Current context and available information
            
        Returns:
            Dictionary containing prediction details
        """
        print(f"🔮 Azure Planner: Predicting outcome for step {plan_step.get('step_id', 'unknown')}...")
        
        prompt = PREDICT_PROMPT.format(
            plan_step=plan_step.get("description", ""),
            current_context=context.get("current_context", ""),
            available_info=context.get("available_info", "")
        )
        
        response, token_info = self._call_azure_openai(prompt)
        
        try:
            prediction_data = json.loads(response)
            
            print(f"✅ Prediction completed with {prediction_data.get('confidence_level', 'unknown')} confidence")
            
            return {
                "success": True,
                "prediction": prediction_data,
                "token_usage": token_info
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Prediction parsing error: {e}")
            
            # Fallback prediction
            fallback_prediction = {
                "primary_prediction": "The step will likely return some relevant information",
                "success_scenario": "Relevant entities or relationships will be discovered",
                "partial_scenario": "Some useful information will be found but may be incomplete",
                "failure_scenario": "No relevant information will be discovered",
                "confidence_level": "low",
                "key_factors": ["Data availability", "Query specificity", "Entity coverage"]
            }
            
            return {
                "success": False,
                "prediction": fallback_prediction,
                "error": str(e),
                "token_usage": token_info
            }
    
    def select_relations(self, entity_name: str, entity_id: str, 
                        available_relations: List[str], plan_step: Dict[str, Any], 
                        question: str) -> Dict[str, Any]:
        """
        Select the most relevant relations for exploration
        
        Args:
            entity_name: Name of the entity
            entity_id: ID of the entity
            available_relations: List of available relations
            plan_step: Current plan step
            question: Original question
            
        Returns:
            Dictionary containing selected relations and reasoning
        """
        print(f"🎯 Azure Planner: Selecting relations for {entity_name}...")
        
        prompt = RELATION_SELECTION_PROMPT.format(
            plan_step=plan_step.get("description", ""),
            entity_name=entity_name,
            entity_id=entity_id,
            relations=", ".join(available_relations),
            question=question
        )
        
        response, token_info = self._call_azure_openai(prompt)
        
        try:
            selection_data = json.loads(response)
            selected_relations = selection_data.get("selected_relations", [])
            
            # Validate selected relations exist in available relations
            valid_relations = [rel for rel in selected_relations if rel in available_relations]
            
            if not valid_relations and available_relations:
                # Fallback: select first few relations
                valid_relations = available_relations[:3]
            
            print(f"✅ Selected {len(valid_relations)} relations: {valid_relations}")
            
            return {
                "success": True,
                "selected_relations": valid_relations,
                "reasoning": selection_data.get("selection_reasoning", ""),
                "backup_relations": selection_data.get("backup_relations", []),
                "token_usage": token_info
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Relation selection parsing error: {e}")
            
            # Fallback: select first few relations
            fallback_relations = available_relations[:3] if available_relations else []
            
            return {
                "success": False,
                "selected_relations": fallback_relations,
                "reasoning": "Fallback selection due to parsing error",
                "error": str(e),
                "token_usage": token_info
            }
    
    def think_and_evaluate(self, prediction: Dict[str, Any], observation: str, 
                          plan_step: Dict[str, Any], memory_summary: Dict[str, Any],
                          question: str) -> Dict[str, Any]:
        """
        Analyze results and determine next action
        
        Args:
            prediction: Previous prediction
            observation: Actual observation from execution
            plan_step: Current plan step
            memory_summary: Summary of current memory state
            question: Original question
            
        Returns:
            Dictionary containing analysis and next action decision
        """
        print(f"🤔 Azure Planner: Analyzing results and determining next action...")
        
        prompt = THINK_AND_EVALUATE_PROMPT.format(
            prediction=str(prediction),
            observation=observation,
            plan_step=plan_step.get("description", ""),
            memory_summary=str(memory_summary),
            question=question
        )
        
        response, token_info = self._call_azure_openai(prompt)
        
        try:
            evaluation_data = json.loads(response)
            next_action = evaluation_data.get("next_action", "PROCEED")
            
            # Validate next action
            valid_actions = ["PROCEED", "CORRECT_PATH", "REPLAN", "FINISH"]
            if next_action not in valid_actions:
                next_action = "PROCEED"
            
            print(f"✅ Analysis complete. Next action: {next_action}")
            print(f"   Reasoning: {evaluation_data.get('reasoning', 'No reasoning provided')}")
            
            return {
                "success": True,
                "evaluation": evaluation_data,
                "next_action": next_action,
                "token_usage": token_info
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Evaluation parsing error: {e}")
            
            # Fallback evaluation
            fallback_evaluation = {
                "prediction_accuracy": "unknown",
                "step_success": "partial_success",
                "information_gained": observation[:100] + "..." if len(observation) > 100 else observation,
                "answer_feasibility": "partial",
                "next_action": "PROCEED",
                "reasoning": "Fallback decision due to parsing error",
                "confidence": "low"
            }
            
            return {
                "success": False,
                "evaluation": fallback_evaluation,
                "next_action": "PROCEED",
                "error": str(e),
                "token_usage": token_info
            }
    
    def replan(self, question: str, failed_plan: List[Dict[str, Any]], 
              failure_reason: str, memory_summary: Dict[str, Any],
              previous_attempts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new strategic plan based on previous failures
        
        Args:
            question: Original question
            failed_plan: The plan that failed
            failure_reason: Reason for failure
            memory_summary: Current memory state
            previous_attempts: Previous planning attempts
            
        Returns:
            Dictionary containing new plan and strategy
        """
        print(f"🔄 Azure Planner: Creating new strategic plan...")
        
        prompt = REPLAN_PROMPT.format(
            question=question,
            failed_plan=str(failed_plan),
            failure_reason=failure_reason,
            memory_summary=str(memory_summary),
            previous_attempts=str(previous_attempts or [])
        )
        
        response, token_info = self._call_azure_openai(prompt)
        
        try:
            replan_data = json.loads(response)
            new_plan = replan_data.get("new_plan", [])
            
            # Validate new plan structure
            for i, step in enumerate(new_plan):
                if "step_id" not in step:
                    step["step_id"] = i + 1
                if "description" not in step:
                    step["description"] = f"Revised step {i + 1}"
            
            print(f"✅ Replanning successful: {len(new_plan)} steps")
            for step in new_plan:
                print(f"   Step {step['step_id']}: {step['description']}")
            
            return {
                "success": True,
                "new_plan": new_plan,
                "strategy_change": replan_data.get("strategy_change", ""),
                "lessons_learned": replan_data.get("lessons_learned", ""),
                "token_usage": token_info
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Replanning parsing error: {e}")
            
            # Fallback: create a simplified plan
            fallback_plan = [
                {"step_id": 1, "description": f"Re-examine the question from a different angle: {question}"},
                {"step_id": 2, "description": "Focus on the most essential entities and relationships"},
                {"step_id": 3, "description": "Synthesize available information for the best possible answer"}
            ]
            
            return {
                "success": False,
                "new_plan": fallback_plan,
                "strategy_change": "Simplified approach due to parsing error",
                "error": str(e),
                "token_usage": token_info
            }
    
    def generate_final_answer(self, question: str, knowledge_summary: Dict[str, Any],
                            reasoning_paths: List[Dict[str, Any]],
                            exploration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate the final answer based on gathered information
        
        Args:
            question: Original question
            knowledge_summary: Summary of gathered knowledge
            reasoning_paths: Successful reasoning paths
            exploration_history: History of exploration steps
            
        Returns:
            Dictionary containing final answer and supporting information
        """
        print(f"📝 Azure Planner: Generating final answer...")
        
        prompt = FINAL_ANSWER_PROMPT.format(
            question=question,
            knowledge_summary=str(knowledge_summary),
            reasoning_paths=str(reasoning_paths),
            exploration_history=str(exploration_history)
        )
        
        response, token_info = self._call_azure_openai(prompt)
        
        try:
            answer_data = json.loads(response)
            
            print(f"✅ Final answer generated")
            print(f"   Answer: {answer_data.get('final_answer', 'No answer provided')}")
            print(f"   Confidence: {answer_data.get('confidence', 'unknown')}")
            
            return {
                "success": True,
                "answer_data": answer_data,
                "token_usage": token_info
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Answer generation parsing error: {e}")
            
            # Fallback answer
            fallback_answer = {
                "final_answer": "Unable to generate a complete answer due to processing limitations.",
                "confidence": "low",
                "supporting_evidence": ["Limited information available"],
                "reasoning_chain": "Attempted systematic exploration but encountered technical difficulties",
                "limitations": "Answer generation was impaired by parsing errors"
            }
            
            return {
                "success": False,
                "answer_data": fallback_answer,
                "error": str(e),
                "token_usage": token_info
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics for the planner
        
        Returns:
            Dictionary containing usage statistics
        """
        return {
            "total_llm_calls": self.total_llm_calls,
            "total_tokens_used": self.total_tokens_used,
            "average_tokens_per_call": self.total_tokens_used / max(1, self.total_llm_calls),
            "deployment_id": self.deployment_id
        }
