"""
Enhanced PPoGA Prompts Module

This module contains all LLM prompts used in the PPoGA system, 
based on the original papers and GitHub implementations.
All prompts are written in English following the academic standards.
"""

# 1. Plan Decomposition Prompt (Based on PLAN-AND-ACT paper)
DECOMPOSE_PLAN_PROMPT = """You are a strategic planner for complex question answering over knowledge graphs. Your task is to decompose a complex question into a sequence of executable sub-goals that can be systematically addressed.

Given the question below, analyze its structure and create a step-by-step plan that breaks down the problem into manageable components. Each step should be specific, actionable, and logically ordered.

Question: {question}

Please provide your response in the following JSON format:
{{
    "plan": [
        {{"step_id": 1, "description": "First actionable step"}},
        {{"step_id": 2, "description": "Second actionable step"}},
        {{"step_id": 3, "description": "Third actionable step"}}
    ],
    "rationale": "Explanation of why this decomposition strategy was chosen and how it addresses the question's complexity"
}}

Guidelines:
- Each step should focus on a single, well-defined sub-goal
- Steps should build upon each other logically
- Consider the knowledge graph structure and typical entity-relation patterns
- Aim for 2-5 steps depending on question complexity"""

# 2. Prediction Prompt (Based on PreAct paper)
PREDICT_PROMPT = """You are tasked with predicting the outcome of executing a specific plan step in a knowledge graph question answering system.

Current Plan Step: {plan_step}
Current Context: {current_context}
Available Information: {available_info}

Based on the current step and context, predict what might happen when this step is executed. Consider multiple scenarios including success, partial success, and failure cases.

Please provide your prediction in the following JSON format:
{{
    "primary_prediction": "Most likely outcome when executing this step",
    "success_scenario": "What would happen if the step succeeds completely",
    "partial_scenario": "What would happen if the step succeeds partially",
    "failure_scenario": "What would happen if the step fails",
    "confidence_level": "high|medium|low",
    "key_factors": ["Factor 1 that might influence the outcome", "Factor 2", "Factor 3"]
}}

Consider:
- The nature of knowledge graph queries and typical result patterns
- Potential ambiguities or missing information
- Common failure modes in entity and relation discovery"""

# 3. Relation Selection Prompt (Based on Plan-on-Graph paper)
RELATION_SELECTION_PROMPT = """You are a knowledge graph exploration specialist. Given an entity and a set of available relations, select the most relevant relations that will help achieve the current plan step.

Current Plan Step: {plan_step}
Target Entity: {entity_name} (ID: {entity_id})
Available Relations: {relations}
Question Context: {question}

Your task is to select the most promising relations to explore, considering:
1. Relevance to the current plan step
2. Likelihood of containing useful information
3. Typical knowledge graph patterns

Please respond in the following JSON format:
{{
    "selected_relations": ["relation1", "relation2", "relation3"],
    "selection_reasoning": "Detailed explanation of why these relations were chosen",
    "expected_information": "What type of information you expect to find through these relations",
    "backup_relations": ["alternative1", "alternative2"]
}}

Guidelines:
- Select 1-3 most relevant relations
- Prioritize relations that directly address the plan step
- Consider both direct and indirect relevance
- Provide backup options in case primary selections fail"""

# 4. Thinking and Evaluation Prompt (Enhanced from Plan-on-Graph)
THINK_AND_EVALUATE_PROMPT = """You are an analytical reasoner evaluating the results of a knowledge graph exploration step. Compare the predicted outcomes with actual observations and determine the next course of action.

Prediction: {prediction}
Actual Observation: {observation}
Current Plan Step: {plan_step}
Memory Summary: {memory_summary}
Question: {question}

Analyze the situation by comparing predictions with reality and evaluate whether sufficient progress has been made toward answering the question.

Please provide your analysis in the following JSON format:
{{
    "prediction_accuracy": "How well did the prediction match reality (accurate|partially_accurate|inaccurate)",
    "step_success": "Whether the current step achieved its goal (success|partial_success|failure)",
    "information_gained": "Summary of new information obtained",
    "answer_feasibility": "Can the question be answered with current information (yes|partial|no)",
    "next_action": "PROCEED|CORRECT_PATH|REPLAN|FINISH",
    "reasoning": "Detailed explanation of the analysis and decision",
    "confidence": "Confidence in the next action decision (high|medium|low)"
}}

Next Action Guidelines:
- PROCEED: Move to next plan step (current step successful, more steps needed)
- CORRECT_PATH: Retry current step with different approach (step failed but plan is sound)
- REPLAN: Create new plan (current plan is fundamentally flawed)
- FINISH: Generate final answer (sufficient information gathered)"""

# 5. Strategic Replanning Prompt (Based on PLAN-AND-ACT paper)
REPLAN_PROMPT = """You are a strategic replanner for knowledge graph question answering. The current plan has encountered significant obstacles and needs to be revised.

Original Question: {question}
Failed Plan: {failed_plan}
Failure Analysis: {failure_reason}
Accumulated Knowledge: {memory_summary}
Previous Attempts: {previous_attempts}

Based on the failures and lessons learned, create a new strategic approach that avoids previous pitfalls and leverages the knowledge already gathered.

Please provide your new plan in the following JSON format:
{{
    "new_plan": [
        {{"step_id": 1, "description": "Revised first step"}},
        {{"step_id": 2, "description": "Revised second step"}},
        {{"step_id": 3, "description": "Revised third step"}}
    ],
    "strategy_change": "How this plan differs from the previous approach",
    "lessons_learned": "Key insights from the previous failure",
    "risk_mitigation": "How this plan addresses previous failure points",
    "confidence_rationale": "Why this new approach is more likely to succeed"
}}

Replanning Principles:
- Learn from previous failures without repeating them
- Leverage any useful information already gathered
- Consider alternative question interpretation if needed
- Ensure each step is more robust and specific than before"""

# 6. Final Answer Generation Prompt (Enhanced from Plan-on-Graph)
FINAL_ANSWER_PROMPT = """You are tasked with generating the final answer to a complex question based on all the information gathered through systematic knowledge graph exploration.

Original Question: {question}
Gathered Knowledge: {knowledge_summary}
Reasoning Paths: {reasoning_paths}
Exploration History: {exploration_history}

Synthesize all available information to provide a comprehensive and accurate answer to the original question.

Please provide your response in the following JSON format:
{{
    "final_answer": "Direct answer to the question",
    "confidence": "high|medium|low",
    "supporting_evidence": ["Evidence 1", "Evidence 2", "Evidence 3"],
    "reasoning_chain": "Step-by-step logical reasoning that led to this answer",
    "alternative_interpretations": "Any alternative ways to interpret the question or answer",
    "limitations": "Any limitations or uncertainties in the answer"
}}

Answer Quality Guidelines:
- Be direct and specific in addressing the question
- Ground your answer in the evidence gathered
- Acknowledge any uncertainties or limitations
- Provide clear reasoning for your conclusions
- Consider alternative interpretations if the question was ambiguous"""

# 7. Entity Filtering Prompt (Based on Plan-on-Graph paper)
ENTITY_FILTERING_PROMPT = """You are an entity relevance assessor for knowledge graph question answering. Given a list of discovered entities, identify those most relevant to the current plan step and overall question.

Question: {question}
Current Plan Step: {plan_step}
Discovered Entities: {entities}
Context: {context}

Evaluate each entity's relevance and select the most promising ones for further exploration.

Please respond in the following JSON format:
{{
    "selected_entities": [
        {{"entity_id": "id1", "entity_name": "name1", "relevance_score": 0.9, "reason": "why this entity is relevant"}},
        {{"entity_id": "id2", "entity_name": "name2", "relevance_score": 0.8, "reason": "why this entity is relevant"}}
    ],
    "filtering_criteria": "What criteria were used to assess relevance",
    "rejected_entities": ["entity_id3", "entity_id4"],
    "rejection_reasons": "Why certain entities were filtered out"
}}

Filtering Guidelines:
- Prioritize entities directly related to the question topic
- Consider entities that might lead to the answer
- Remove obviously irrelevant or generic entities
- Maintain 3-7 entities for manageable exploration
- Score relevance from 0.0 to 1.0"""

# 8. Memory Update Prompt (Enhanced from Plan-on-Graph)
MEMORY_UPDATE_PROMPT = """You are a knowledge organizer responsible for updating the system's memory with newly discovered information.

Question: {question}
Sub-objectives: {sub_objectives}
New Information: {new_information}
Current Memory State: {existing_memory}
Exploration Context: {context}

Organize and integrate the new information into the existing knowledge structure, updating progress on sub-objectives.

Please provide the updated memory in the following JSON format:
{{
    "updated_objectives": {{
        "objective_1": {{"status": "completed|in_progress|not_started", "findings": "what was discovered"}},
        "objective_2": {{"status": "completed|in_progress|not_started", "findings": "what was discovered"}}
    }},
    "key_discoveries": ["Discovery 1", "Discovery 2", "Discovery 3"],
    "entity_relationships": [
        {{"source": "entity1", "relation": "relation_type", "target": "entity2", "confidence": 0.9}}
    ],
    "progress_assessment": "Overall progress toward answering the question",
    "next_priorities": ["What should be explored next", "Secondary priority"]
}}

Memory Organization Principles:
- Maintain clear connections between discoveries and objectives
- Track confidence levels for uncertain information
- Identify gaps that still need to be filled
- Organize information for easy retrieval and reasoning"""

# 9. Backtracking Decision Prompt (Based on Plan-on-Graph paper)
BACKTRACK_JUDGMENT_PROMPT = """You are a path evaluation specialist determining whether the current exploration path should be abandoned in favor of backtracking to a previous state.

Current Entities: {current_entities}
Memory State: {memory_state}
Question: {question}
Exploration History: {exploration_history}
Current Plan Step: {current_step}

Assess whether the current path is likely to lead to a solution or if backtracking would be more productive.

Please provide your assessment in the following JSON format:
{{
    "should_backtrack": true/false,
    "backtrack_reasoning": "Detailed explanation of the backtracking decision",
    "current_path_assessment": "Evaluation of the current exploration path",
    "alternative_paths": ["Path 1 description", "Path 2 description"],
    "backtrack_target": "Which previous state or entity to return to",
    "expected_benefit": "What backtracking is expected to achieve"
}}

Backtracking Criteria:
- Current path has reached a dead end
- Alternative paths show higher promise
- Current entities are too far from the question topic
- Exploration has become too narrow or too broad
- Previous promising leads were abandoned prematurely"""

# 10. Execution Summary Prompt (For Executor reporting to Planner)
EXECUTION_SUMMARY_PROMPT = """You are an execution reporter summarizing knowledge graph query results for strategic decision-making.

Executed Query Information: {query_info}
Raw Results: {raw_results}
Current Plan Step: {plan_step}
Query Context: {context}

Summarize the execution results in a format that enables effective strategic decision-making.

Please provide your summary in the following JSON format:
{{
    "execution_summary": "Concise overview of what was accomplished",
    "entities_discovered": {{"count": 5, "types": ["type1", "type2"]}},
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "data_quality": "Assessment of result quality and completeness",
    "relevance_assessment": "How relevant the results are to the current plan step",
    "recommended_next_steps": ["Suggestion 1", "Suggestion 2"],
    "potential_issues": ["Issue 1", "Issue 2"]
}}

Reporting Guidelines:
- Focus on actionable insights rather than raw data
- Highlight unexpected or particularly relevant findings
- Assess data quality and completeness
- Suggest logical next steps based on results
- Flag any issues that might affect planning"""
