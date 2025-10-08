#!/usr/bin/env python3
"""
LangGraph-based PPoGA Implementation

This module implements PPoGA using LangGraph's StateGraph for clear workflow modeling
with Azure OpenAI integration.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ppoga.robust_azure_planner import RobustAzurePPoGAPlanner


class PPoGAState(TypedDict):
    """State definition for PPoGA LangGraph workflow"""
    # Input
    question: str
    topic_entities: Dict[str, str]
    
    # Planning
    current_plan: List[Dict[str, Any]]
    current_step_id: int
    plan_rationale: str
    replan_count: int
    
    # Execution
    current_prediction: Dict[str, Any]
    current_observation: str
    execution_log: List[Dict[str, Any]]
    
    # Knowledge
    discovered_entities: List[str]
    exploration_history: List[Dict[str, Any]]
    
    # Control flow
    next_action: str
    iteration_count: int
    max_iterations: int
    
    # Results
    final_answer: str
    confidence: str
    supporting_evidence: List[str]
    
    # Statistics
    llm_calls: int
    kg_queries: int
    execution_time: float


class LangGraphPPoGA:
    """
    LangGraph-based PPoGA implementation with Azure OpenAI
    """
    
    def __init__(self, azure_config: Dict[str, Any]):
        """Initialize LangGraph PPoGA system"""
        self.azure_config = azure_config
        
        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_config["api_base"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_deployment=azure_config["deployment_id"],
            temperature=0.1,
            max_tokens=2048
        )
        
        # Initialize planner for complex operations
        self.planner = RobustAzurePPoGAPlanner(azure_config)
        
        # Mock executor for testing
        self.executor = EnhancedMockKnowledgeGraphExecutor()
        
        # Build the state graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for PPoGA workflow"""
        
        # Create the state graph
        workflow = StateGraph(PPoGAState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("predict", self._predict_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("think", self._think_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("replan", self._replan_node)
        workflow.add_node("final_answer", self._final_answer_node)
        
        # Set entry point
        workflow.set_entry_point("plan")
        
        # Add edges
        workflow.add_edge("plan", "predict")
        workflow.add_edge("predict", "execute")
        workflow.add_edge("execute", "think")
        workflow.add_edge("think", "evaluate")
        
        # Conditional edges from evaluate
        workflow.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {
                "predict": "predict",      # PROCEED to next step
                "execute": "execute",      # CORRECT_PATH - retry execution
                "replan": "replan",        # REPLAN - create new plan
                "final_answer": "final_answer",  # FINISH - generate answer
                "end": END                 # Maximum iterations reached
            }
        )
        
        # Replan goes back to predict
        workflow.add_edge("replan", "predict")
        
        # Final answer ends the workflow
        workflow.add_edge("final_answer", END)
        
        return workflow.compile()
    
    def _plan_node(self, state: PPoGAState) -> Dict[str, Any]:
        """Initial planning node"""
        print(f"📋 LangGraph Node: Planning")
        
        plan_result = self.planner.decompose_plan(state["question"])
        
        return {
            "current_plan": plan_result["plan"],
            "plan_rationale": plan_result["rationale"],
            "current_step_id": 0,
            "replan_count": 0,
            "llm_calls": state["llm_calls"] + 1
        }
    
    def _predict_node(self, state: PPoGAState) -> Dict[str, Any]:
        """Prediction node"""
        print(f"🔮 LangGraph Node: Predicting (Step {state['current_step_id'] + 1})")
        
        if state["current_step_id"] >= len(state["current_plan"]):
            return {"next_action": "FINISH"}
        
        current_step = state["current_plan"][state["current_step_id"]]
        
        context = {
            "current_context": f"Executing step {state['current_step_id'] + 1} of {len(state['current_plan'])}",
            "available_info": f"Discovered {len(state['discovered_entities'])} entities so far"
        }
        
        prediction_result = self.planner.predict_step_outcome(current_step, context)
        
        return {
            "current_prediction": prediction_result["prediction"],
            "llm_calls": state["llm_calls"] + 1
        }
    
    def _execute_node(self, state: PPoGAState) -> Dict[str, Any]:
        """Execution node"""
        print(f"🤖 LangGraph Node: Executing")
        
        if state["current_step_id"] >= len(state["current_plan"]):
            return {"current_observation": "No more steps to execute"}
        
        current_step = state["current_plan"][state["current_step_id"]]
        
        if state["topic_entities"]:
            entity_id = list(state["topic_entities"].keys())[0]
            entity_name = state["topic_entities"][entity_id]
            
            # Get available relations
            available_relations = self.executor.get_available_relations(entity_id, entity_name)
            
            # Select relations
            relation_result = self.planner.select_relations(
                entity_name, entity_id, available_relations, current_step, state["question"]
            )
            selected_relations = relation_result["selected_relations"]
            
            # Execute step
            execution_result = self.executor.execute_step(
                entity_id, entity_name, selected_relations, current_step["description"]
            )
            
            new_entities = state["discovered_entities"] + execution_result["discovered_entities"]
            
            return {
                "current_observation": execution_result["observation"],
                "discovered_entities": new_entities,
                "kg_queries": state["kg_queries"] + 1,
                "llm_calls": state["llm_calls"] + 1
            }
        else:
            return {
                "current_observation": "No topic entities available for exploration",
                "llm_calls": state["llm_calls"] + 1
            }
    
    def _think_node(self, state: PPoGAState) -> Dict[str, Any]:
        """Thinking and analysis node"""
        print(f"🤔 LangGraph Node: Thinking")
        
        # Update exploration history
        new_history_entry = {
            "step_id": state["current_step_id"] + 1,
            "prediction": state["current_prediction"],
            "observation": state["current_observation"],
            "timestamp": time.time()
        }
        
        updated_history = state["exploration_history"] + [new_history_entry]
        
        return {
            "exploration_history": updated_history
        }
    
    def _evaluate_node(self, state: PPoGAState) -> Dict[str, Any]:
        """Evaluation and decision node"""
        print(f"⚖️ LangGraph Node: Evaluating")
        
        if state["current_step_id"] >= len(state["current_plan"]):
            return {"next_action": "FINISH"}
        
        current_step = state["current_plan"][state["current_step_id"]]
        
        memory_summary = {
            "current_step": state["current_step_id"],
            "total_steps": len(state["current_plan"]),
            "entities_discovered": len(set(state["discovered_entities"])),
            "execution_cycles": len(state["execution_log"]),
            "replan_count": state["replan_count"]
        }
        
        evaluation_result = self.planner.think_and_evaluate(
            state["current_prediction"],
            state["current_observation"],
            current_step,
            memory_summary,
            state["question"]
        )
        
        next_action = evaluation_result["next_action"]
        
        # Update execution log
        log_entry = {
            "step_id": state["current_step_id"] + 1,
            "prediction": state["current_prediction"],
            "observation": state["current_observation"],
            "evaluation": evaluation_result["evaluation"],
            "next_action": next_action,
            "timestamp": time.time()
        }
        
        updated_log = state["execution_log"] + [log_entry]
        
        # Update step counter if proceeding
        new_step_id = state["current_step_id"]
        if next_action == "PROCEED":
            new_step_id += 1
        
        return {
            "next_action": next_action,
            "execution_log": updated_log,
            "current_step_id": new_step_id,
            "iteration_count": state["iteration_count"] + 1,
            "llm_calls": state["llm_calls"] + 1
        }
    
    def _replan_node(self, state: PPoGAState) -> Dict[str, Any]:
        """Replanning node"""
        print(f"🔄 LangGraph Node: Replanning")
        
        memory_summary = {
            "current_step": state["current_step_id"],
            "total_steps": len(state["current_plan"]),
            "entities_discovered": len(set(state["discovered_entities"])),
            "execution_cycles": len(state["execution_log"]),
            "replan_count": state["replan_count"]
        }
        
        replan_result = self.planner.replan(
            state["question"],
            state["current_plan"],
            "Current plan not yielding sufficient progress",
            memory_summary,
            state["exploration_history"]
        )
        
        return {
            "current_plan": replan_result["new_plan"],
            "plan_rationale": replan_result.get("strategy_change", ""),
            "current_step_id": 0,
            "replan_count": state["replan_count"] + 1,
            "llm_calls": state["llm_calls"] + 1
        }
    
    def _final_answer_node(self, state: PPoGAState) -> Dict[str, Any]:
        """Final answer generation node"""
        print(f"📝 LangGraph Node: Generating Final Answer")
        
        memory_summary = {
            "question": state["question"],
            "entities_discovered": len(set(state["discovered_entities"])),
            "execution_cycles": len(state["execution_log"]),
            "replan_count": state["replan_count"]
        }
        
        answer_result = self.planner.generate_final_answer(
            state["question"],
            memory_summary,
            [],  # reasoning_paths
            state["exploration_history"]
        )
        
        answer_data = answer_result["answer_data"]
        
        return {
            "final_answer": answer_data["final_answer"],
            "confidence": answer_data["confidence"],
            "supporting_evidence": answer_data.get("supporting_evidence", []),
            "llm_calls": state["llm_calls"] + 1
        }
    
    def _should_continue(self, state: PPoGAState) -> str:
        """Determine the next node based on evaluation results"""
        
        # Check iteration limit
        if state["iteration_count"] >= state["max_iterations"]:
            print(f"⏰ Maximum iterations ({state['max_iterations']}) reached")
            return "end"
        
        next_action = state.get("next_action", "PROCEED")
        
        if next_action == "PROCEED":
            # Check if we've completed all steps
            if state["current_step_id"] >= len(state["current_plan"]):
                return "final_answer"
            else:
                return "predict"
        elif next_action == "CORRECT_PATH":
            return "execute"
        elif next_action == "REPLAN":
            # Limit replanning attempts
            if state["replan_count"] >= 2:
                print("⚠️ Maximum replans reached, finishing")
                return "final_answer"
            else:
                return "replan"
        elif next_action == "FINISH":
            return "final_answer"
        else:
            # Default to continuing
            return "predict"
    
    def run(self, question: str, topic_entities: Dict[str, str], 
            max_iterations: int = 8) -> Dict[str, Any]:
        """
        Run the LangGraph PPoGA workflow
        
        Args:
            question: Question to answer
            topic_entities: Initial topic entities
            max_iterations: Maximum iterations
            
        Returns:
            Results dictionary
        """
        start_time = time.time()
        
        print(f"🚀 LangGraph PPoGA Starting")
        print(f"   Question: {question}")
        print(f"   Topic Entities: {list(topic_entities.values())}")
        print(f"   Max Iterations: {max_iterations}")
        
        # Initialize state
        initial_state = PPoGAState(
            question=question,
            topic_entities=topic_entities,
            current_plan=[],
            current_step_id=0,
            plan_rationale="",
            replan_count=0,
            current_prediction={},
            current_observation="",
            execution_log=[],
            discovered_entities=[],
            exploration_history=[],
            next_action="PROCEED",
            iteration_count=0,
            max_iterations=max_iterations,
            final_answer="",
            confidence="",
            supporting_evidence=[],
            llm_calls=0,
            kg_queries=0,
            execution_time=0.0
        )
        
        try:
            # Run the graph
            print(f"\n{'='*60}")
            print(f"LANGGRAPH PPOGA WORKFLOW EXECUTION")
            print(f"{'='*60}")
            
            final_state = self.graph.invoke(initial_state)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Get statistics
            planner_stats = self.planner.get_statistics()
            executor_stats = self.executor.get_statistics()
            
            result = {
                "success": True,
                "question": question,
                "answer": final_state["final_answer"],
                "confidence": final_state["confidence"],
                "supporting_evidence": final_state["supporting_evidence"],
                "execution_time": execution_time,
                "iterations": final_state["iteration_count"],
                "statistics": {
                    "llm_calls": final_state["llm_calls"],
                    "kg_queries": final_state["kg_queries"],
                    "entities_discovered": len(set(final_state["discovered_entities"])),
                    "replan_count": final_state["replan_count"],
                    "planner_stats": planner_stats,
                    "executor_stats": executor_stats
                },
                "execution_log": final_state["execution_log"][-3:],  # Last 3 entries
                "workflow_type": "LangGraph StateGraph"
            }
            
            print(f"\n{'='*60}")
            print(f"LANGGRAPH PPOGA EXECUTION COMPLETED")
            print(f"{'='*60}")
            print(f"Question: {question}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Execution Time: {execution_time:.2f} seconds")
            print(f"Iterations: {final_state['iteration_count']}")
            print(f"LLM Calls: {final_state['llm_calls']}")
            print(f"KG Queries: {final_state['kg_queries']}")
            print(f"Entities Discovered: {len(set(final_state['discovered_entities']))}")
            
            return result
            
        except Exception as e:
            print(f"❌ LangGraph PPoGA execution error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "execution_time": time.time() - start_time,
                "workflow_type": "LangGraph StateGraph"
            }


class EnhancedMockKnowledgeGraphExecutor:
    """Enhanced Mock Knowledge Graph Executor (same as in robust implementation)"""
    
    def __init__(self):
        self.query_count = 0
        
        # Comprehensive mock data
        self.mock_data = {
            "taylor_swift": {
                "name": "Taylor Swift",
                "type": "music.artist",
                "relations": {
                    "music.artist.track": [
                        "Shake It Off", "Anti-Hero", "Blank Space", "Love Story", 
                        "We Are Never Ever Getting Back Together", "Look What You Made Me Do",
                        "Bad Blood", "22", "I Knew You Were Trouble", "Style"
                    ],
                    "music.artist.album": [
                        "1989", "Folklore", "Evermore", "Midnights", "Reputation", 
                        "Red", "Speak Now", "Fearless", "Taylor Swift"
                    ],
                    "award.award_winner.awards_won": [
                        "American Music Award for Favorite Pop/Rock Female Artist 2019",
                        "American Music Award for Artist of the Year 2020",
                        "American Music Award for Favorite Music Video - Anti-Hero 2022",
                        "American Music Award for Favorite Pop Song - Anti-Hero 2022",
                        "Grammy Award for Album of the Year - Folklore",
                        "Grammy Award for Song of the Year - cardigan"
                    ]
                }
            }
        }
    
    def get_available_relations(self, entity_id: str, entity_name: str) -> List[str]:
        """Get available relations for an entity"""
        print(f"🔍 Enhanced Mock Executor: Getting relations for {entity_name}")
        
        if "taylor" in entity_name.lower() or "swift" in entity_name.lower():
            relations = [
                "music.artist.track", 
                "music.artist.album", 
                "award.award_winner.awards_won",
                "people.person.profession"
            ]
        else:
            relations = ["type.object.type", "common.topic.notable_types"]
        
        print(f"   Found {len(relations)} relations: {relations}")
        return relations
    
    def execute_step(self, entity_id: str, entity_name: str, selected_relations: List[str],
                    plan_step_description: str) -> Dict[str, Any]:
        """Execute a knowledge graph exploration step"""
        print(f"🤖 Enhanced Mock Executor: Exploring {entity_name} with relations {selected_relations}")
        
        self.query_count += 1
        
        discovered_entities = []
        relation_results = {}
        
        for relation in selected_relations:
            entities = []
            
            if "taylor" in entity_name.lower() and relation == "music.artist.track":
                entities = self.mock_data["taylor_swift"]["relations"]["music.artist.track"]
            elif "taylor" in entity_name.lower() and relation == "award.award_winner.awards_won":
                entities = self.mock_data["taylor_swift"]["relations"]["award.award_winner.awards_won"]
            elif "taylor" in entity_name.lower() and relation == "music.artist.album":
                entities = self.mock_data["taylor_swift"]["relations"]["music.artist.album"]
            else:
                entities = [f"Mock_Entity_{i}" for i in range(1, 3)]
            
            discovered_entities.extend(entities)
            relation_results[relation] = entities
        
        unique_entities = list(set(discovered_entities))
        total_entities = len(unique_entities)
        
        if total_entities > 0:
            award_entities = [e for e in unique_entities if "American Music Award" in e]
            song_entities = [e for e in unique_entities if e in self.mock_data["taylor_swift"]["relations"]["music.artist.track"]]
            
            observation_parts = []
            if award_entities:
                observation_parts.append(f"Found {len(award_entities)} American Music Award connections")
            if song_entities:
                observation_parts.append(f"Identified {len(song_entities)} Taylor Swift songs")
            if not observation_parts:
                observation_parts.append(f"Discovered {total_entities} related entities")
            
            observation = f"Successfully explored {entity_name}: " + ", ".join(observation_parts)
            
            if award_entities and song_entities:
                observation += ". Notable finding: 'Anti-Hero' appears to have won American Music Awards."
        else:
            observation = f"No entities found for {entity_name} with the selected relations"
        
        execution_info = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "relations_explored": selected_relations,
            "relation_results": relation_results,
            "total_entities_found": total_entities,
            "query_count": 1
        }
        
        print(f"   ✅ Execution complete: {observation}")
        
        return {
            "observation": observation,
            "execution_info": execution_info,
            "discovered_entities": unique_entities
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        return {
            "total_queries": self.query_count,
            "mock_mode": True,
            "data_sources": ["Taylor Swift discography", "American Music Awards data"]
        }


def main():
    """Main function for testing LangGraph PPoGA"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph PPoGA System")
    parser.add_argument("--question", type=str,
                       default="Which of Taylor Swift's songs has won American Music Awards?",
                       help="Question to answer")
    parser.add_argument("--max_iterations", type=int, default=6,
                       help="Maximum number of iterations")
    parser.add_argument("--output_file", type=str, default="langgraph_ppoga_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Azure OpenAI configuration
    azure_config = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
        "api_version": os.environ.get("OPENAI_API_VERSION"),
        "deployment_id": os.environ.get("DEPLOYMENT_ID"),
        "temperature": 0.1,
        "max_tokens": 2048
    }
    
    # Test topic entities
    topic_entities = {
        "m.0dl567": "Taylor Swift"
    }
    
    try:
        # Initialize and run LangGraph PPoGA
        ppoga_system = LangGraphPPoGA(azure_config)
        result = ppoga_system.run(args.question, topic_entities, args.max_iterations)
        
        # Save results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📁 Results saved to {args.output_file}")
        
        if result["success"]:
            print(f"\n🎉 LangGraph PPoGA completed successfully!")
        else:
            print(f"\n❌ LangGraph PPoGA encountered errors")
            
    except Exception as e:
        print(f"❌ Main execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
