#!/usr/bin/env python3
"""
PPoGA v2: Predictive Plan-on-Graph with Action

Main execution script that combines PoG's proven SPARQL engine
with PPoGA's predictive planning and enhanced memory system.

This implementation bridges:
- PoG's reliable knowledge graph exploration
- PreAct's predictive reasoning
- PLAN-AND-ACT's dynamic replanning
- Enhanced 3-layer memory architecture

Usage:
    python main_ppoga_v2.py --question "Who directed The Godfather?" --max_iterations 8
    python main_ppoga_v2.py --dataset cwq --max_iterations 10
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from config import PPoGAConfig
from ppoga_v2.predictive_planner import PredictivePlanner
from ppoga_v2.enhanced_executor import EnhancedExecutor
from ppoga_v2.enhanced_memory import PPoGAMemory


def run_ppoga_v2(
    question: str, topic_entities: Dict[str, str], config: PPoGAConfig
) -> Dict[str, Any]:
    """
    Run PPoGA v2 system combining PoG's SPARQL engine with predictive planning

    Args:
        question: The question to answer
        topic_entities: Initial topic entities {entity_id: entity_name}
        config: PPoGA configuration

    Returns:
        Dictionary containing results and statistics
    """
    start_time = time.time()

    print(f"🚀 PPoGA v2 System Starting")
    print(f"   Question: {question}")
    print(f"   Topic Entities: {list(topic_entities.values())}")
    print(f"   Using Model: {config.model}")
    print(f"   SPARQL Endpoint: {config.sparql_endpoint}")

    # Initialize components
    memory = PPoGAMemory(question)
    planner = PredictivePlanner(config.get_llm_config())
    executor = EnhancedExecutor(config.get_kg_config())
    args = config.to_args_namespace()

    try:
        # =====================================================================
        # STEP 1: STRATEGIC PLAN DECOMPOSITION
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"STEP 1: STRATEGIC PLAN DECOMPOSITION")
        print(f"{'='*60}")

        context = {"initial_entities": list(topic_entities.values())}
        plan_result = planner.decompose_plan(question, context)

        # Log LLM interaction for plan decomposition (PPoGA Step 1)
        memory.log_llm_interaction(
            call_type="step1_plan_decomposition",
            prompt=f"Question: {question}\nContext: {context}",
            response=str(plan_result),
            success=plan_result["success"],
            metadata={
                "iteration": 0,
                "ppoga_step": "1_plan_decomposition",
                "step_description": "Strategic Plan Decomposition",
            },
        )

        memory.increment_llm_calls()

        if not plan_result["success"]:
            print(f"⚠️ Plan decomposition had issues, using fallback plan")

        memory.update_plan(plan_result["plan"], plan_result["rationale"])

        # =====================================================================
        # STEP 2: PREDICTIVE EXECUTION LOOP
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"STEP 2: PREDICTIVE EXECUTION LOOP")
        print(f"{'='*60}")

        iteration = 0
        while iteration < config.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            current_step = memory.get_current_step()
            if not current_step:
                print("✅ All plan steps completed")
                break

            print(f"Current Step {current_step.step_id}: {current_step.description}")

            # 2.1 PREDICTION PHASE (PreAct)
            context = {
                "current_context": f"Executing step {current_step.step_id} of {len(memory.strategy['overall_plan'])}",
                "available_info": f"Discovered {len(memory.knowledge['discovered_entities'])} entities so far",
                "memory_context": memory.get_context_for_llm(),
            }

            prediction_result = planner.predict_step_outcome(
                current_step.__dict__, context
            )

            # Log prediction interaction (PPoGA Step 2)
            memory.log_llm_interaction(
                call_type="step2_prediction",
                prompt=f"Step: {current_step.__dict__}\nContext: {context}",
                response=str(prediction_result),
                success=prediction_result["success"],
                metadata={
                    "iteration": iteration,
                    "step_id": current_step.step_id,
                    "ppoga_step": "2_prediction",
                    "step_description": "Predictive Processing - Outcome Prediction",
                },
            )

            memory.increment_llm_calls()

            if not prediction_result["success"]:
                print("⚠️ Prediction failed, using fallback")

            prediction = prediction_result["prediction"]

            # 2.2 EXECUTION PHASE (PPoGA Step 3 - Action Execution by Enhanced Executor)
            if topic_entities:
                # Log Step 3 start
                memory.log_llm_interaction(
                    call_type="step3_execution_start",
                    prompt=f"PPoGA Step 3 - Execution Phase\nEntities to explore: {list(topic_entities.values())[:3]}\nPlan Step: {current_step.__dict__}",
                    response=f"Starting execution for {len(list(topic_entities.items())[:3])} entities",
                    success=True,
                    metadata={
                        "iteration": iteration,
                        "step_id": current_step.step_id,
                        "ppoga_step": "3_execution",
                        "step_description": "Action Execution - SPARQL queries and knowledge graph exploration",
                        "entities_count": len(list(topic_entities.items())[:3]),
                    },
                )

                # Execute for each available entity
                step_observations = []
                step_discovered_entities = {}

                for entity_id, entity_name in list(topic_entities.items())[
                    :3
                ]:  # Limit to 3 entities per step
                    print(f"   Exploring entity: {entity_name}")

                    execution_result = executor.execute_step(
                        entity_id=entity_id,
                        entity_name=entity_name,
                        plan_step=current_step.__dict__,
                        args=args,
                    )

                    if execution_result["success"]:
                        step_observations.append(execution_result["observation"])
                        step_discovered_entities.update(
                            execution_result["discovered_entities"]
                        )

                        # Update memory with exploration results
                        for relation, rel_results in execution_result[
                            "exploration_results"
                        ].items():
                            memory.add_exploration_result(
                                entity_id, relation, rel_results["entity_ids"]
                            )

                        memory.increment_sparql_queries()
                    else:
                        step_observations.append(
                            f"Execution failed for {entity_name}: {execution_result.get('error', 'Unknown error')}"
                        )

                # Combine observations
                combined_observation = "\n".join(step_observations)

                # Update memory with discovered entities
                memory.add_discovered_entities(step_discovered_entities)

                # Update topic entities for next iteration
                topic_entities.update(step_discovered_entities)

            else:
                combined_observation = "No topic entities available for exploration"

            # 2.3 OBSERVATION, THINKING AND EVALUATION PHASE (PPoGA Steps 4 & 6)
            memory_context = memory.get_context_for_llm()

            # Note: Step 3 (Execution) is handled by enhanced_executor above

            evaluation_result = planner.think_and_evaluate(
                prediction=prediction,
                observation=combined_observation,
                plan_step=current_step.__dict__,
                memory_context=memory_context,
            )

            # Log as PPoGA Step 4: Observation and Thought
            memory.log_llm_interaction(
                call_type="step4_observation_thought",
                prompt=f"PPoGA Step 4 - Observation & Thought Analysis\nPrediction: {prediction}\nObservation: {combined_observation}\nStep: {current_step.__dict__}\nMemory: {memory_context}",
                response=str(
                    evaluation_result.get(
                        "step4_observation_thought",
                        evaluation_result.get("evaluation", {}),
                    )
                ),
                success=evaluation_result.get("success", True),
                metadata={
                    "iteration": iteration,
                    "step_id": current_step.step_id,
                    "ppoga_step": "4_observation_thought",
                    "step_description": "Observation and Thought - Compare prediction with actual results",
                },
            )

            # Log as PPoGA Step 6: Evaluation and Two-Level Self-Correction
            memory.log_llm_interaction(
                call_type="step6_evaluation_correction",
                prompt=f"PPoGA Step 6 - Evaluation & Two-Level Self-Correction\nThought Analysis: {evaluation_result.get('step4_observation_thought', {})}\nNext Action Decision Required",
                response=str(
                    evaluation_result.get(
                        "step6_evaluation_correction",
                        {
                            "next_action": evaluation_result.get(
                                "next_action", "PROCEED"
                            )
                        },
                    )
                ),
                success=evaluation_result.get("success", True),
                metadata={
                    "iteration": iteration,
                    "step_id": current_step.step_id,
                    "ppoga_step": "6_evaluation_correction",
                    "step_description": "Evaluation and Two-Level Self-Correction - Decide next action",
                    "next_action": evaluation_result.get("next_action", "PROCEED"),
                },
            )

            memory.increment_llm_calls()

            evaluation = evaluation_result["evaluation"]
            next_action = evaluation_result["next_action"]

            # 2.4 MEMORY UPDATE
            memory.add_execution_cycle(
                step_id=current_step.step_id,
                prediction=prediction,
                action={"entity_exploration": list(topic_entities.keys())},
                observation=combined_observation,
                thought=evaluation,
            )

            # 2.5 ACTION DECISION
            if next_action == "PROCEED":
                if not memory.advance_step():
                    print("✅ All steps completed")
                    break
            elif next_action == "CORRECT_PATH":
                print("🔄 Correcting path - retrying current step")
                continue
            elif next_action == "REPLAN":
                print("🔄 Replanning strategy")
                replan_result = planner.replan(
                    question=question,
                    failed_plan=[
                        step.__dict__ for step in memory.strategy["overall_plan"]
                    ],
                    failure_reason="Current plan not yielding sufficient progress",
                    memory_context=memory_context,
                )

                # Log replanning interaction
                memory.log_llm_interaction(
                    call_type="replan",
                    prompt=f"Question: {question}\nFailed Plan: {[step.__dict__ for step in memory.strategy['overall_plan']]}\nFailure Reason: Current plan not yielding sufficient progress\nMemory: {memory_context}",
                    response=str(replan_result),
                    success=replan_result.get("success", False),
                    metadata={
                        "iteration": iteration,
                        "replan_count": memory.strategy["replan_count"],
                    },
                )

                memory.increment_llm_calls()

                if replan_result["success"]:
                    memory.replan(
                        new_plan=replan_result["new_plan"],
                        new_rationale=replan_result.get("strategy_change", ""),
                        reason="Strategic replanning triggered",
                    )
                    continue
                else:
                    print("❌ Replanning failed, continuing with original plan")
                    if not memory.advance_step():
                        break
            elif next_action == "FINISH":
                print("✅ Sufficient information gathered")
                break
            else:
                print(f"⚠️ Unknown action {next_action}, proceeding")
                if not memory.advance_step():
                    break

        # =====================================================================
        # STEP 3: FINAL ANSWER GENERATION
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"STEP 3: FINAL ANSWER GENERATION")
        print(f"{'='*60}")

        answer_result = planner.generate_final_answer(
            question=question,
            memory_context=memory.get_context_for_llm(),
            reasoning_chains=memory.knowledge["reasoning_chains"],
        )

        # Log final answer generation (PPoGA Step 5: Memory Integration + Answer Synthesis)
        memory.log_llm_interaction(
            call_type="step5_final_answer_generation",
            prompt=f"PPoGA Step 5 - Final Answer Generation\nQuestion: {question}\nMemory Context: {memory.get_context_for_llm()}\nReasoning Chains: {memory.knowledge['reasoning_chains']}",
            response=str(answer_result),
            success=answer_result.get("success", True),
            metadata={
                "ppoga_step": "5_final_answer_generation",
                "step_description": "Memory Integration and Final Answer Synthesis",
                "final_step": True,
                "total_iterations": iteration,
                "total_entities_discovered": len(
                    memory.knowledge.get("discovered_entities", {})
                ),
                "total_relations_explored": len(
                    memory.knowledge.get("relations_explored", [])
                ),
            },
        )

        memory.increment_llm_calls()

        # =====================================================================
        # COMPILE RESULTS
        # =====================================================================
        end_time = time.time()
        execution_time = end_time - start_time

        planner_stats = planner.get_statistics()
        executor_stats = executor.get_statistics()
        memory_summary = memory.get_memory_summary()

        result = {
            "success": True,
            "question": question,
            "answer": answer_result["answer_data"]["final_answer"],
            "confidence": answer_result["answer_data"]["confidence"],
            "supporting_evidence": answer_result["answer_data"].get(
                "supporting_evidence", []
            ),
            "reasoning": answer_result["answer_data"].get("reasoning_chain", ""),
            "limitations": answer_result["answer_data"].get("limitations", ""),
            "execution_time": execution_time,
            "iterations": iteration,
            "statistics": {
                "memory_summary": memory_summary,
                "planner_stats": planner_stats,
                "executor_stats": executor_stats,
                "total_llm_calls": memory.stats["llm_calls"],
                "total_sparql_queries": memory.stats["sparql_queries"],
            },
        }

        print(f"\n{'='*60}")
        print(f"🎉 PPoGA v2 EXECUTION COMPLETED")
        print(f"{'='*60}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Execution Time: {execution_time:.2f}s")
        print(f"Iterations: {iteration}")
        print(f"LLM Calls: {memory_summary['llm_calls']}")
        print(f"SPARQL Queries: {memory_summary['sparql_queries']}")
        print(f"Entities Discovered: {memory_summary['entities_discovered']}")

        return result

    except Exception as e:
        print(f"❌ PPoGA v2 execution failed: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "iterations": iteration if "iteration" in locals() else 0,
        }


def load_sample_entities(question: str) -> Dict[str, str]:
    """
    Load sample topic entities (placeholder for dataset integration)

    In a full implementation, this would:
    1. Parse the question to identify entities
    2. Resolve them to Freebase IDs
    3. Return the mapping
    """
    # Sample entity mappings for common questions (verified Freebase IDs)
    sample_mappings = {
        "godfather": {"m.07g1sm": "The Godfather"},  # Verified from SPARQL
        "titanic": {"m.0dr_4": "Titanic"},  # 1997 James Cameron film
        "taylor swift": {"m.0dl567": "Taylor Swift"},
        "obama": {"m.02mjmr": "Barack Obama"},
        "france": {"m.0f8l9c": "France"},
        "microsoft": {"m.04593p": "Microsoft"},
    }

    question_lower = question.lower()
    for key, entities in sample_mappings.items():
        if key in question_lower:
            return entities

    # Default fallback
    return {"m.0sample": "Sample Entity"}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="PPoGA v2: Predictive Plan-on-Graph with Action"
    )
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cwq",
        help="Dataset to use (cwq, webqsp, grailqa)",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=10, help="Maximum iterations"
    )
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum search depth")
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create configuration from environment variables first
    config = PPoGAConfig.from_env()

    # Override with command line arguments
    config.question = args.question or config.question
    config.dataset = args.dataset
    config.max_iterations = args.max_iterations
    config.max_depth = args.max_depth
    config.model = args.model
    config.output_dir = args.output_dir
    config.verbose = args.verbose

    # Validate configuration
    if not config.validate():
        print("❌ Configuration validation failed")
        return

    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    if args.question:
        # Single question mode
        print(f"Running PPoGA v2 for question: {args.question}")

        # Load topic entities (placeholder)
        topic_entities = load_sample_entities(args.question)

        # Run PPoGA v2
        result = run_ppoga_v2(args.question, topic_entities, config)

        # Save results
        timestamp = int(time.time())
        output_file = os.path.join(
            config.output_dir, f"ppoga_v2_result_{timestamp}.json"
        )

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Results saved to: {output_file}")

    else:
        print("❌ Please provide a question using --question")
        print(
            "Example: python main_ppoga_v2.py --question 'Who directed The Godfather?'"
        )


if __name__ == "__main__":
    main()
