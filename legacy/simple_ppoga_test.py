"""
PPoGA-on-PoG System - Simplified Synchronous Version
For testing and development purposes
"""

import os
import sys
import time
import json
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

class SimplePPoGASystem:
    """Simplified PPoGA system for testing"""
    
    def __init__(self, azure_config: Optional[Dict[str, str]] = None):
        self.mock_mode = True
        
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
        
        print(f"🚀 Simple PPoGA System initialized")
    
    def _load_azure_config(self) -> Dict[str, str]:
        """Load Azure configuration from environment or .env file"""
        try:
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        except FileNotFoundError:
            pass
        
        config = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_type": "azure",
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            "deployment_id": os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        }
        
        return config
    
    def solve_question(self, question: str, max_steps: int = 5) -> Dict[str, Any]:
        """Main PPoGA solving pipeline - simplified version"""
        print(f"\n🎯 PPoGA Question: {question}")
        print("="*80)
        
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
            
            if not plan_result.get("success", False):
                return self._create_error_result(f"Planning failed: {plan_result.get('error')}")
            
            plan_steps = plan_result.get("plan", [])
            print(f"✅ Created plan with {len(plan_steps)} steps")
            
            for i, step in enumerate(plan_steps, 1):
                print(f"   Step {i}: {step.get('description', 'No description')}")
            
            # Update memory with plan
            self.memory.update_plan(plan_steps, "Initial predictive plan", "Decompose and execute")
            
            # =================================================================
            # PHASE 2: SIMPLIFIED EXECUTION
            # =================================================================
            print("\n⚡ PHASE 2: Simplified Execution")
            print("-" * 40)
            
            final_answer = None
            
            for step_count in range(1, min(len(plan_steps) + 1, max_steps + 1)):
                current_step = self.memory.get_current_step()
                
                if not current_step:
                    print("❌ No current step available")
                    break
                
                print(f"\n🔄 Step {step_count}: {current_step['description']}")
                
                # Simple execution simulation
                time.sleep(0.5)  # Simulate processing
                
                # Mock step execution based on question
                if "directed" in question.lower() and "godfather" in question.lower():
                    if step_count == 1:
                        observation = "Found entity: m.03f4l (The Godfather movie)"
                        success = True
                    elif step_count == 2:
                        observation = "Found director: m.02rdx8 (Francis Ford Coppola)"
                        final_answer = "Francis Ford Coppola directed The Godfather"
                        success = True
                    else:
                        observation = f"Completed step {step_count}"
                        success = True
                else:
                    observation = f"Mock execution result for step {step_count}"
                    success = True
                    if step_count == len(plan_steps):
                        final_answer = f"Mock answer for: {question}"
                
                # Record execution cycle
                self.memory.add_execution_cycle(
                    step_id=step_count,
                    prediction={"summary": f"Expected result for step {step_count}"},
                    action=current_step['description'],
                    observation=observation,
                    thought={"reasoning": f"Step {step_count} completed", "confidence": 0.8},
                    success=success
                )
                
                if success:
                    print(f"✅ Step {step_count} completed: {observation[:100]}...")
                    
                    if final_answer:
                        print(f"🎯 Found final answer: {final_answer}")
                        self.memory.strategic["status"] = PlanStatus.FINISHED.value
                        break
                    
                    # Advance to next step
                    if not self.memory.advance_step():
                        break
                else:
                    print(f"❌ Step {step_count} failed")
                    break
            
            # =================================================================
            # PHASE 3: RESULT COMPILATION
            # =================================================================
            print("\n📊 PHASE 3: Result Compilation")
            print("-" * 40)
            
            execution_time = time.time() - self.start_time
            self.memory.add_execution_time(execution_time)
            
            result = self._compile_final_result(final_answer, execution_time)
            
            self.memory.print_memory_status()
            print(f"\n🏁 Simple PPoGA completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ PPoGA system error: {str(e)}")
            return self._create_error_result(f"System error: {str(e)}")
    
    def _compile_final_result(self, final_answer: Optional[str], execution_time: float) -> Dict[str, Any]:
        """Compile final execution result"""
        memory_summary = self.memory.get_memory_summary()
        reasoning_chain = self.memory.get_reasoning_chain()
        
        return {
            "question": self.question,
            "answer": final_answer or "No definitive answer found",
            "success": final_answer is not None,
            "execution_time": execution_time,
            "memory_summary": memory_summary,
            "reasoning_chain": reasoning_chain,
            "statistics": {
                "steps_executed": len(reasoning_chain),
                "entities_discovered": memory_summary["knowledge_summary"]["entities_discovered"],
                "llm_calls": memory_summary["statistics"]["llm_calls"],
                "success_rate": memory_summary["execution_summary"]["success_rate"],
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary"""
        return {
            "question": self.question,
            "answer": f"Error: {error_message}",
            "success": False,
            "execution_time": time.time() - self.start_time if self.start_time > 0 else 0,
            "error": error_message,
            "memory_summary": self.memory.get_memory_summary() if self.memory else {},
            "statistics": {}
        }

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python simple_ppoga_test.py \"<question>\"")
        print("Example: python simple_ppoga_test.py \"Who directed The Godfather?\"")
        sys.exit(1)
    
    question = sys.argv[1]
    
    # Initialize system
    system = SimplePPoGASystem()
    
    # Solve question
    result = system.solve_question(question, max_steps=5)
    
    # Print results
    print("\n" + "="*80)
    print("🎯 FINAL RESULT")
    print("="*80)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    
    if result.get("statistics"):
        stats = result["statistics"]
        print(f"\n📊 Statistics:")
        print(f"   Steps Executed: {stats.get('steps_executed', 0)}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.1%}")
    
    print("="*80)

if __name__ == "__main__":
    main()
