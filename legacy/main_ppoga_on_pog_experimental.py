"""
PPoGA-on-PoG Main System
Combines Plan-on-Graph (PoG) with Predictive PreAct capabilities

Usage:
    poetry run python -m ppoga_project.main_ppoga_on_pog "Who is the spouse of the director of The Godfather?"
"""

import os
import sys
import time
import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ppoga_project.pog_base.azure_llm import AzureLLMClient
from ppoga_project.pog_base.freebase_func import (
    id2entity_name_or_type,
    entity_search,
    relation_search_prune,
)
from ppoga_project.ppoga_core.predictive_planner import PredictivePlanner
from ppoga_project.ppoga_core.enhanced_executor import EnhancedKGExecutor
from ppoga_project.ppoga_core.enhanced_memory import ThreeLayerMemory, PlanStatus


class PPoGASystem:
    """
    PPoGA (Predictive Plan-on-Graph with Action) System

    Combines:
    - PoG's proven SPARQL-based knowledge graph querying
    - PreAct's predictive reasoning capabilities
    - Enhanced 3-layer memory architecture
    - Self-correction and adaptation mechanisms
    """

    def __init__(
        self, azure_config: Optional[Dict[str, str]] = None, mock_mode: bool = False
    ):
        """
        Initialize PPoGA system

        Args:
            azure_config: Azure OpenAI configuration
            mock_mode: Use mock KG for testing (default: False)
        """
        self.mock_mode = mock_mode

        # Initialize Azure LLM client
        if azure_config is None:
            azure_config = self._load_azure_config()

        self.llm_client = AzureLLMClient(azure_config)

        # Initialize core components
        self.planner = PredictivePlanner(azure_config)
        self.executor = EnhancedKGExecutor(azure_config)

        # Current session
        self.memory: Optional[ThreeLayerMemory] = None
        self.question = ""
        self.start_time = 0.0

        print(
            f"🚀 PPoGA System initialized (Mock Mode: {'ON' if mock_mode else 'OFF'})"
        )

    def _load_azure_config(self) -> Dict[str, str]:
        """Load Azure configuration from environment or .env file"""
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        config = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT", "ktc-nego-0919"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        }

        if not config["api_key"] or not config["endpoint"]:
            print("⚠️  Warning: Azure configuration not found in environment")
            print("   Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")

        return config

    def solve_question(
        self, question: str, max_steps: int = 10, save_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Main PPoGA solving pipeline

        Args:
            question: Natural language question to solve
            max_steps: Maximum planning steps
            save_memory: Save memory to file

        Returns:
            Dictionary with answer and execution details
        """
        print(f"\n🎯 PPoGA Question: {question}")
        print("=" * 80)

        self.question = question
        self.start_time = time.time()

        # Initialize 3-layer memory
        self.memory = ThreeLayerMemory(question)

        try:
            # =================================================================
            # PHASE 1: PREDICTIVE PLANNING
            # =================================================================
            print("\n📋 PHASE 1: Predictive Planning")
            print("-" * 40)

            plan_result = self.planner.decompose_plan_with_prediction(
                question=question, context={}
            )

            if not plan_result["success"]:
                return self._create_error_result(
                    f"Planning failed: {plan_result['error']}"
                )

            print(f"✅ Created plan with {len(plan_result['plan'])} steps")
            self.memory.print_memory_status()

            # =================================================================
            # PHASE 2: PREDICTIVE EXECUTION
            # =================================================================
            print("\n⚡ PHASE 2: Predictive Execution")
            print("-" * 40)

            step_count = 0
            final_answer = None

            while step_count < max_steps and self.memory.strategic["status"] not in [
                PlanStatus.FINISHED.value,
                PlanStatus.FAILED.value,
            ]:

                step_count += 1
                current_step = self.memory.get_current_step()

                if not current_step:
                    print("❌ No current step available")
                    break

                print(f"\n🔄 Step {step_count}: {current_step['description']}")

                # Execute step with prediction
                step_result = self._execute_predictive_step(
                    current_step, step_count
                )

                if step_result["success"]:
                    print(f"✅ Step {step_count} completed successfully")

                    # Check if we have a final answer
                    if step_result.get("final_answer"):
                        final_answer = step_result["final_answer"]
                        self.memory.strategic["status"] = PlanStatus.FINISHED.value
                        break

                    # Advance to next step
                    if not self.memory.advance_step():
                        break

                else:
                    print(
                        f"❌ Step {step_count} failed: {step_result.get('error', 'Unknown error')}"
                    )

                    # Attempt recovery
                    recovery_result = self._attempt_recovery(
                        step_result, step_count
                    )

                    if not recovery_result["success"]:
                        self.memory.strategic["status"] = PlanStatus.FAILED.value
                        break

            # =================================================================
            # PHASE 3: RESULT COMPILATION
            # =================================================================
            print("\n📊 PHASE 3: Result Compilation")
            print("-" * 40)

            execution_time = time.time() - self.start_time
            self.memory.add_execution_time(execution_time)

            result = self._compile_final_result(final_answer, execution_time)

            # Save memory if requested
            if save_memory:
                memory_file = f"memory_snapshot_{int(self.start_time)}.json"
                self.memory.save_to_file(memory_file)
                result["memory_file"] = memory_file

            self.memory.print_memory_status()
            print(f"\n🏁 PPoGA completed in {execution_time:.2f}s")

            return result

        except Exception as e:
            print(f"❌ PPoGA system error: {str(e)}")
            return self._create_error_result(f"System error: {str(e)}")

    async def _execute_predictive_step(
        self, step: Dict[str, Any], step_count: int
    ) -> Dict[str, Any]:
        """Execute a single predictive step"""
        try:
            # =================================================================
            # 1. PREDICTION: What do we expect to happen?
            # =================================================================
            print(f"🔮 Predicting step outcome...")

            prediction_result = await self.planner.predict_step_outcome(
                step=step, memory=self.memory
            )

            if not prediction_result["success"]:
                return {
                    "success": False,
                    "error": f"Prediction failed: {prediction_result['error']}",
                }

            prediction = prediction_result["prediction"]
            print(f"   Expected: {prediction.get('summary', 'No summary')}")

            # =================================================================
            # 2. ACTION: Execute the planned action
            # =================================================================
            print(f"🎬 Executing action...")

            action_result = await self.executor.execute_step_with_correction(
                step=step, prediction=prediction, memory=self.memory
            )

            if not action_result.success:
                return {
                    "success": False,
                    "error": f"Action failed: {action_result.error}",
                }

            observation = action_result.result
            print(
                f"   Observed: {observation[:100]}..."
                if len(observation) > 100
                else f"   Observed: {observation}"
            )

            # =================================================================
            # 3. THOUGHT: Analyze and learn from the result
            # =================================================================
            print(f"🤔 Analyzing result...")

            thought_result = await self.planner.think_and_evaluate(
                step=step,
                prediction=prediction,
                observation=observation,
                memory=self.memory,
            )

            if not thought_result["success"]:
                return {
                    "success": False,
                    "error": f"Analysis failed: {thought_result['error']}",
                }

            thought = thought_result["thought"]
            print(f"   Analysis: {thought.get('reasoning', 'No reasoning')}")

            # =================================================================
            # 4. MEMORY UPDATE: Record the complete cycle
            # =================================================================
            self.memory.add_execution_cycle(
                step_id=step_count,
                prediction=prediction,
                action=step["description"],
                observation=observation,
                thought=thought,
                success=True,
            )

            # Check for final answer
            final_answer = None
            if thought.get("has_final_answer", False):
                final_answer = thought.get("final_answer", "")
                print(f"🎯 Found final answer: {final_answer}")

            return {
                "success": True,
                "prediction": prediction,
                "observation": observation,
                "thought": thought,
                "final_answer": final_answer,
            }

        except Exception as e:
            print(f"❌ Step execution error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _attempt_recovery(
        self, failed_result: Dict[str, Any], step_count: int
    ) -> Dict[str, Any]:
        """Attempt to recover from step failure"""
        try:
            print(f"🔧 Attempting recovery for step {step_count}...")

            # Record correction attempt
            self.memory.add_correction_attempt(
                step_id=step_count,
                attempt_type="step_recovery",
                reason=failed_result.get("error", "Step failed"),
                result="Attempting recovery",
            )

            # Check if we should replan
            failure_rate = self.memory.execution["failed_cycles"] / max(
                self.memory.execution["total_cycles"], 1
            )

            if failure_rate > 0.3:  # If >30% failure rate, consider replanning
                print("🔄 High failure rate detected, considering replanning...")

                replan_result = await self.planner.replan_if_needed(
                    current_failure=failed_result.get("error", ""), memory=self.memory
                )

                if replan_result["should_replan"]:
                    print("🔄 Replanning initiated")
                    return {"success": True, "action": "replanned"}

            # Simple retry for now
            print("🔂 Retrying current step...")
            return {"success": True, "action": "retry"}

        except Exception as e:
            print(f"❌ Recovery failed: {str(e)}")
            return {"success": False, "error": f"Recovery failed: {str(e)}"}

    def _compile_final_result(
        self, final_answer: Optional[str], execution_time: float
    ) -> Dict[str, Any]:
        """Compile final execution result"""
        memory_summary = self.memory.get_memory_summary()
        reasoning_chain = self.memory.get_reasoning_chain()
        exploration_summary = self.memory.get_exploration_summary()

        return {
            "question": self.question,
            "answer": final_answer or "No definitive answer found",
            "success": final_answer is not None,
            "execution_time": execution_time,
            "memory_summary": memory_summary,
            "reasoning_chain": reasoning_chain,
            "exploration_summary": exploration_summary,
            "statistics": {
                "steps_executed": len(reasoning_chain),
                "entities_discovered": memory_summary["knowledge_summary"][
                    "entities_discovered"
                ],
                "llm_calls": memory_summary["statistics"]["llm_calls"],
                "kg_queries": memory_summary["statistics"]["kg_queries"],
                "success_rate": memory_summary["execution_summary"]["success_rate"],
                "replan_count": memory_summary["strategic_summary"]["replan_count"],
            },
        }

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            "question": self.question,
            "answer": f"Error: {error_message}",
            "success": False,
            "execution_time": (
                time.time() - self.start_time if self.start_time > 0 else 0
            ),
            "error": error_message,
            "memory_summary": self.memory.get_memory_summary() if self.memory else {},
            "statistics": {},
        }


# =================================================================
# COMMAND LINE INTERFACE
# =================================================================


async def main():
    """Command line interface for PPoGA"""
    if len(sys.argv) < 2:
        print('Usage: python -m ppoga_project.main_ppoga_on_pog "<question>"')
        print(
            'Example: python -m ppoga_project.main_ppoga_on_pog "Who is the spouse of the director of The Godfather?"'
        )
        sys.exit(1)

    question = sys.argv[1]

    # Check for mock mode flag
    mock_mode = "--mock" in sys.argv

    # Initialize system
    system = PPoGASystem(mock_mode=mock_mode)

    # Solve question
    result = await system.solve_question(question, max_steps=10, save_memory=True)

    # Print results
    print("\n" + "=" * 80)
    print("🎯 FINAL RESULT")
    print("=" * 80)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
    print(f"Execution Time: {result['execution_time']:.2f}s")

    if result.get("statistics"):
        stats = result["statistics"]
        print(f"\n📊 Statistics:")
        print(f"   Steps Executed: {stats.get('steps_executed', 0)}")
        print(f"   Entities Discovered: {stats.get('entities_discovered', 0)}")
        print(f"   LLM Calls: {stats.get('llm_calls', 0)}")
        print(f"   KG Queries: {stats.get('kg_queries', 0)}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.1%}")
        if stats.get("replan_count", 0) > 0:
            print(f"   Replans: {stats.get('replan_count', 0)}")

    if result.get("memory_file"):
        print(f"\n💾 Memory snapshot saved to: {result['memory_file']}")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
