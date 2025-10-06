#!/usr/bin/env python3
"""
Robust Azure OpenAI PPoGA Main Execution Script

This script runs the PPoGA system using Azure OpenAI API with enhanced error handling
and robust JSON parsing capabilities.
"""

import os
import json
import time
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ppoga.robust_azure_planner import RobustAzurePPoGAPlanner


class EnhancedMockKnowledgeGraphExecutor:
    """
    Enhanced Mock Knowledge Graph Executor with more realistic data
    """
    
    def __init__(self):
        self.query_count = 0
        
        # More comprehensive mock data
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
                    ],
                    "people.person.profession": ["Singer-songwriter", "Record producer", "Actress"]
                }
            },
            "american_music_awards": {
                "name": "American Music Awards",
                "type": "award.award_ceremony",
                "relations": {
                    "award.award_ceremony.awards": [
                        "Favorite Pop/Rock Female Artist",
                        "Artist of the Year",
                        "Favorite Music Video",
                        "Favorite Pop Song"
                    ]
                }
            }
        }
    
    def get_available_relations(self, entity_id: str, entity_name: str) -> List[str]:
        """Get available relations for an entity"""
        print(f"🔍 Enhanced Mock Executor: Getting relations for {entity_name}")
        
        # More realistic relation discovery
        if "taylor" in entity_name.lower() or "swift" in entity_name.lower():
            relations = [
                "music.artist.track", 
                "music.artist.album", 
                "award.award_winner.awards_won",
                "people.person.profession",
                "music.artist.genre",
                "people.person.date_of_birth"
            ]
        elif "award" in entity_name.lower():
            relations = [
                "award.award.winners", 
                "award.award.category", 
                "award.award.year",
                "award.award_ceremony.awards"
            ]
        elif "song" in entity_name.lower() or "track" in entity_name.lower():
            relations = [
                "music.recording.artist",
                "music.recording.releases",
                "award.award_winning_work.awards_won"
            ]
        else:
            relations = [
                "type.object.type", 
                "common.topic.notable_types",
                "common.topic.description"
            ]
        
        print(f"   Found {len(relations)} relations: {relations}")
        return relations
    
    def execute_step(self, entity_id: str, entity_name: str, selected_relations: List[str],
                    plan_step_description: str) -> Dict[str, Any]:
        """Execute a knowledge graph exploration step with realistic results"""
        print(f"🤖 Enhanced Mock Executor: Exploring {entity_name} with relations {selected_relations}")
        
        self.query_count += 1
        
        # More sophisticated mock query execution
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
                
            elif relation == "award.award.winners":
                entities = ["Taylor Swift", "Beyonce", "Drake", "Ariana Grande", "Ed Sheeran"]
                
            elif relation == "award.award_ceremony.awards":
                entities = self.mock_data["american_music_awards"]["relations"]["award.award_ceremony.awards"]
                
            else:
                # Generic mock results based on relation type
                if "music" in relation:
                    entities = [f"Music_Entity_{i}" for i in range(1, 4)]
                elif "award" in relation:
                    entities = [f"Award_Entity_{i}" for i in range(1, 3)]
                else:
                    entities = [f"Generic_Entity_{i}" for i in range(1, 3)]
            
            discovered_entities.extend(entities)
            relation_results[relation] = entities
        
        # Create more detailed observation summary
        total_entities = len(discovered_entities)
        unique_entities = list(set(discovered_entities))  # Remove duplicates
        unique_count = len(unique_entities)
        
        if total_entities > 0:
            # Identify award-related entities specifically
            award_entities = [e for e in unique_entities if "American Music Award" in e]
            song_entities = [e for e in unique_entities if e in self.mock_data["taylor_swift"]["relations"]["music.artist.track"]]
            
            observation_parts = []
            if award_entities:
                observation_parts.append(f"Found {len(award_entities)} American Music Award connections")
            if song_entities:
                observation_parts.append(f"Identified {len(song_entities)} Taylor Swift songs")
            if not observation_parts:
                observation_parts.append(f"Discovered {unique_count} related entities")
            
            observation = f"Successfully explored {entity_name}: " + ", ".join(observation_parts)
            
            # Add specific findings for award-song connections
            if award_entities and song_entities:
                observation += ". Notable finding: 'Anti-Hero' appears to have won American Music Awards for Favorite Music Video and Favorite Pop Song."
                
        else:
            observation = f"No entities found for {entity_name} with the selected relations"
        
        execution_info = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "relations_explored": selected_relations,
            "relation_results": relation_results,
            "total_entities_found": total_entities,
            "unique_entities_found": unique_count,
            "query_count": 1,
            "award_connections": len([e for e in unique_entities if "American Music Award" in e]),
            "song_connections": len([e for e in unique_entities if e in self.mock_data["taylor_swift"]["relations"]["music.artist.track"]])
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


def run_robust_azure_ppoga(question: str, topic_entities: Dict[str, str], 
                          azure_config: Dict[str, Any], max_iterations: int = 8) -> Dict[str, Any]:
    """
    Run the robust PPoGA system with Azure OpenAI
    
    Args:
        question: The question to answer
        topic_entities: Initial topic entities
        azure_config: Azure OpenAI configuration
        max_iterations: Maximum number of iterations
        
    Returns:
        Dictionary containing the results
    """
    start_time = time.time()
    
    print(f"🚀 Robust Azure PPoGA System Starting")
    print(f"   Question: {question}")
    print(f"   Topic Entities: {list(topic_entities.values())}")
    print(f"   Using Azure OpenAI: {azure_config['deployment_id']}")
    print(f"   Max Iterations: {max_iterations}")
    
    # Initialize components
    memory = PPoGAMemory(question)
    planner = RobustAzurePPoGAPlanner(azure_config)
    executor = EnhancedMockKnowledgeGraphExecutor()
    
    try:
        # Step 1: Plan Decomposition
        print(f"\n{'='*60}")
        print(f"STEP 1: PLAN DECOMPOSITION")
        print(f"{'='*60}")
        
        plan_result = planner.decompose_plan(question)
        memory.llm_calls += 1
        
        if not plan_result["success"]:
            print(f"⚠️ Plan decomposition used fallback, but continuing...")
        
        memory.update_plan(plan_result["plan"], plan_result["rationale"])
        
        # Step 2: Execution Loop
        print(f"\n{'='*60}")
        print(f"STEP 2: EXECUTION LOOP")
        print(f"{'='*60}")
        
        iteration = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            current_step = memory.get_current_step()
            if not current_step:
                print("✅ All plan steps completed")
                break
            
            print(f"Current Step {current_step['step_id']}: {current_step['description']}")
            
            # 2.1 Prediction
            context = {
                "current_context": f"Executing step {current_step['step_id']} of {len(memory.current_plan)}",
                "available_info": f"Discovered {len(memory.discovered_entities)} entities so far"
            }
            
            prediction_result = planner.predict_step_outcome(current_step, context)
            memory.current_prediction = prediction_result["prediction"]
            memory.llm_calls += 1
            
            # 2.2 Execution
            if topic_entities:
                entity_id = list(topic_entities.keys())[0]
                entity_name = topic_entities[entity_id]
                
                # Get available relations
                available_relations = executor.get_available_relations(entity_id, entity_name)
                
                # Select relations
                relation_result = planner.select_relations(
                    entity_name, entity_id, available_relations, current_step, question
                )
                selected_relations = relation_result["selected_relations"]
                memory.llm_calls += 1
                
                # Execute step
                execution_result = executor.execute_step(
                    entity_id, entity_name, selected_relations, current_step["description"]
                )
                
                memory.current_observation = execution_result["observation"]
                memory.discovered_entities.extend(execution_result["discovered_entities"])
                memory.kg_queries += execution_result["execution_info"]["query_count"]
                
                # Track successful execution
                if execution_result["execution_info"]["total_entities_found"] > 0:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
            else:
                memory.current_observation = "No topic entities available for exploration"
                consecutive_failures += 1
            
            # 2.3 Thinking and Evaluation
            evaluation_result = planner.think_and_evaluate(
                memory.current_prediction,
                memory.current_observation,
                current_step,
                memory.get_memory_summary(),
                question
            )
            
            evaluation = evaluation_result["evaluation"]
            next_action = evaluation_result["next_action"]
            memory.llm_calls += 1
            
            # Log this execution cycle
            memory.add_execution_log(
                current_step["step_id"],
                memory.current_prediction,
                memory.current_observation,
                evaluation
            )
            
            # 2.4 Action Decision
            if next_action == "PROCEED":
                if not memory.advance_step():
                    print("✅ All steps completed")
                    break
                consecutive_failures = 0  # Reset on successful proceed
                
            elif next_action == "CORRECT_PATH":
                print("🔄 Correcting path - retrying current step")
                if consecutive_failures >= max_consecutive_failures:
                    print("⚠️ Too many consecutive failures, forcing proceed")
                    if not memory.advance_step():
                        break
                continue
                
            elif next_action == "REPLAN":
                print("🔄 Replanning strategy")
                # Only replan if we haven't done it too many times
                if memory.replan_count < 2:
                    replan_result = planner.replan(
                        question,
                        memory.current_plan,
                        "Current plan not yielding sufficient progress",
                        memory.get_memory_summary()
                    )
                    memory.update_plan(replan_result["new_plan"], replan_result.get("strategy_change", ""))
                    memory.replan_count += 1
                    memory.llm_calls += 1
                    consecutive_failures = 0
                else:
                    print("⚠️ Maximum replans reached, forcing proceed")
                    if not memory.advance_step():
                        break
                continue
                
            elif next_action == "FINISH":
                print("✅ Sufficient information gathered")
                break
            else:
                print(f"⚠️ Unknown action {next_action}, proceeding")
                if not memory.advance_step():
                    break
        
        # Step 3: Final Answer Generation
        print(f"\n{'='*60}")
        print(f"STEP 3: FINAL ANSWER GENERATION")
        print(f"{'='*60}")
        
        answer_result = planner.generate_final_answer(
            question,
            memory.get_memory_summary(),
            memory.reasoning_paths,
            memory.exploration_history
        )
        memory.llm_calls += 1
        
        # Compile results
        end_time = time.time()
        execution_time = end_time - start_time
        
        planner_stats = planner.get_statistics()
        executor_stats = executor.get_statistics()
        
        result = {
            "success": True,
            "question": question,
            "answer": answer_result["answer_data"]["final_answer"],
            "confidence": answer_result["answer_data"]["confidence"],
            "supporting_evidence": answer_result["answer_data"].get("supporting_evidence", []),
            "reasoning": answer_result["answer_data"].get("reasoning_chain", ""),
            "limitations": answer_result["answer_data"].get("limitations", ""),
            "execution_time": execution_time,
            "iterations": iteration,
            "statistics": {
                "memory_summary": memory.get_memory_summary(),
                "planner_stats": planner_stats,
                "executor_stats": executor_stats,
                "parsing_success_rate": f"{sum(1 for log in memory.execution_log if 'error' not in str(log))}/{len(memory.execution_log)}"
            },
            "execution_log": memory.execution_log[-5:]  # Last 5 entries to avoid too much data
        }
        
        print(f"\n{'='*60}")
        print(f"ROBUST AZURE PPOGA EXECUTION COMPLETED")
        print(f"{'='*60}")
        print(f"Question: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Iterations: {iteration}")
        print(f"LLM Calls: {planner_stats['total_llm_calls']}")
        print(f"Tokens Used: {planner_stats['total_tokens_used']}")
        print(f"KG Queries: {executor_stats['total_queries']}")
        print(f"Entities Discovered: {len(memory.discovered_entities)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Robust Azure PPoGA execution error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "question": question,
            "execution_time": time.time() - start_time
        }


class PPoGAMemory:
    """Enhanced PPoGA Memory for robust implementation"""
    
    def __init__(self, question: str):
        self.question = question
        self.created_at = time.time()
        
        # Strategic layer
        self.current_plan = []
        self.plan_rationale = ""
        self.current_step_id = 0
        self.replan_count = 0
        
        # Execution layer
        self.execution_log = []
        self.current_prediction = {}
        self.current_observation = ""
        
        # Knowledge layer
        self.discovered_entities = []
        self.exploration_history = []
        self.reasoning_paths = []
        
        # Statistics
        self.llm_calls = 0
        self.kg_queries = 0
    
    def update_plan(self, plan: List[Dict[str, Any]], rationale: str):
        """Update the current plan"""
        self.current_plan = plan
        self.plan_rationale = rationale
        self.current_step_id = 0
    
    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get the current plan step"""
        if self.current_step_id < len(self.current_plan):
            return self.current_plan[self.current_step_id]
        return None
    
    def advance_step(self) -> bool:
        """Advance to the next step"""
        self.current_step_id += 1
        return self.current_step_id < len(self.current_plan)
    
    def add_execution_log(self, step_id: int, prediction: Dict[str, Any], 
                         observation: str, evaluation: Dict[str, Any]):
        """Add an execution log entry"""
        log_entry = {
            "step_id": step_id,
            "prediction": prediction,
            "observation": observation,
            "evaluation": evaluation,
            "timestamp": time.time()
        }
        self.execution_log.append(log_entry)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state"""
        return {
            "question": self.question,
            "current_step": self.current_step_id,
            "total_steps": len(self.current_plan),
            "entities_discovered": len(set(self.discovered_entities)),  # Unique count
            "execution_cycles": len(self.execution_log),
            "replan_count": self.replan_count,
            "llm_calls": self.llm_calls,
            "kg_queries": self.kg_queries
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Robust Azure OpenAI PPoGA System")
    
    parser.add_argument("--question", type=str,
                       default="Which of Taylor Swift's songs has won American Music Awards?",
                       help="Question to answer")
    parser.add_argument("--max_iterations", type=int, default=8,
                       help="Maximum number of iterations")
    parser.add_argument("--output_file", type=str, default="robust_azure_ppoga_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Azure OpenAI configuration
    azure_config = {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "api_base": os.environ.get("OPENAI_API_BASE"),
        "api_version": os.environ.get("OPENAI_API_VERSION"),
        "deployment_id": os.environ.get("DEPLOYMENT_ID"),
        "temperature": 0.1,  # Lower temperature for more consistent responses
        "max_tokens": 2048
    }
    
    # Test topic entities
    topic_entities = {
        "m.0dl567": "Taylor Swift"
    }
    
    try:
        # Run Robust Azure PPoGA
        result = run_robust_azure_ppoga(args.question, topic_entities, azure_config, args.max_iterations)
        
        # Save results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n📁 Results saved to {args.output_file}")
        
        if result["success"]:
            print(f"\n🎉 Robust Azure PPoGA completed successfully!")
            print(f"Final Answer: {result['answer']}")
        else:
            print(f"\n❌ Robust Azure PPoGA encountered errors")
            
    except Exception as e:
        print(f"❌ Main execution error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
