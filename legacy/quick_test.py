"""
Quick PPoGA Test - Mock Only Version
Simple test without external dependencies
"""

import json
import time
from typing import Dict, Any, List


class MockPPoGASystem:
    """Simple mock version for testing basic functionality"""

    def __init__(self):
        self.question = ""
        self.start_time = 0.0

    def solve_question(self, question: str) -> Dict[str, Any]:
        """Mock solve function with predefined responses"""
        print(f"\n🎯 Mock PPoGA Question: {question}")
        print("=" * 80)

        self.question = question
        self.start_time = time.time()

        # Mock planning phase
        print("\n📋 PHASE 1: Predictive Planning")
        print("-" * 40)
        print("✅ Created mock plan with 3 steps")

        # Mock execution phase
        print("\n⚡ PHASE 2: Predictive Execution")
        print("-" * 40)

        # Mock step execution
        steps = [
            "Identify key entities in the question",
            "Search for relevant relationships in knowledge graph",
            "Find and verify the final answer",
        ]

        for i, step in enumerate(steps, 1):
            print(f"\n🔄 Step {i}: {step}")
            time.sleep(0.5)  # Simulate processing
            print(f"✅ Step {i} completed successfully")

        # Mock answer based on question
        if "directed" in question.lower() and "godfather" in question.lower():
            final_answer = "Francis Ford Coppola directed The Godfather"
        elif "spouse" in question.lower() and "director" in question.lower():
            final_answer = "Eleanor Coppola is the spouse of Francis Ford Coppola, the director of The Godfather"
        elif "spielberg" in question.lower():
            final_answer = "Steven Spielberg directed many movies including Jaws, E.T., Jurassic Park, and Schindler's List"
        else:
            final_answer = f"Mock answer for: {question}"

        # Mock result compilation
        print("\n📊 PHASE 3: Result Compilation")
        print("-" * 40)

        execution_time = time.time() - self.start_time

        result = {
            "question": question,
            "answer": final_answer,
            "success": True,
            "execution_time": execution_time,
            "statistics": {
                "steps_executed": len(steps),
                "entities_discovered": 5,
                "llm_calls": 8,
                "kg_queries": 3,
                "success_rate": 1.0,
                "replan_count": 0,
            },
            "memory_summary": {
                "status": "finished",
                "strategic_summary": {
                    "current_step": 3,
                    "total_steps": 3,
                    "replan_count": 0,
                },
                "execution_summary": {
                    "total_cycles": 3,
                    "successful_cycles": 3,
                    "success_rate": 1.0,
                },
                "knowledge_summary": {
                    "entities_discovered": 5,
                    "exploration_paths": 3,
                    "key_findings": 2,
                },
            },
        }

        print(f"\n🏁 Mock PPoGA completed in {execution_time:.2f}s")
        return result


def main():
    """Simple command line interface"""
    import sys

    if len(sys.argv) < 2:
        print('Usage: python quick_test.py "<question>"')
        sys.exit(1)

    question = sys.argv[1]

    # Initialize mock system
    system = MockPPoGASystem()

    # Solve question
    result = system.solve_question(question)

    # Print results
    print("\n" + "=" * 80)
    print("🎯 FINAL RESULT")
    print("=" * 80)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
    print(f"Execution Time: {result['execution_time']:.2f}s")

    stats = result["statistics"]
    print(f"\n📊 Statistics:")
    print(f"   Steps Executed: {stats['steps_executed']}")
    print(f"   Entities Discovered: {stats['entities_discovered']}")
    print(f"   LLM Calls: {stats['llm_calls']}")
    print(f"   KG Queries: {stats['kg_queries']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")

    print("=" * 80)


if __name__ == "__main__":
    main()
