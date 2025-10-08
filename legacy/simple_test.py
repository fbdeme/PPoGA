"""
Simple PPoGA Test Runner
Synchronous version for quick testing
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Simple test without async complexity"""
    if len(sys.argv) < 2:
        print("Usage: python simple_test.py \"<question>\"")
        sys.exit(1)
    
    question = sys.argv[1]
    
    print(f"\n🎯 Simple PPoGA Test: {question}")
    print("="*80)
    
    try:
        # Test imports first
        print("📦 Testing imports...")
        
        from ppoga_project.pog_base.azure_llm import AzureLLMClient
        print("✅ Azure LLM Client imported")
        
        from ppoga_project.pog_base.freebase_func import (
            id2entity_name_or_type,
            entity_search,
            relation_search_prune,
        )
        print("✅ Freebase functions imported")
        
        from ppoga_project.ppoga_core.predictive_planner import PredictivePlanner
        print("✅ Predictive Planner imported")
        
        from ppoga_project.ppoga_core.enhanced_executor import EnhancedKGExecutor
        print("✅ Enhanced KG Executor imported")
        
        from ppoga_project.ppoga_core.enhanced_memory import ThreeLayerMemory, PlanStatus
        print("✅ Enhanced Memory imported")
        
        # Test Azure config
        print("\n🔧 Testing Azure configuration...")
        
        import os
        
        # Load from .env
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        
        azure_config = {
            "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_type": "azure",
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", ""),
            "deployment_id": os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        }
        
        print(f"   Endpoint: {azure_config['api_base']}")
        print(f"   Deployment: {azure_config['deployment_id']}")
        
        # Test components
        print("\n🏗️ Testing component initialization...")
        
        # Test LLM Client
        llm_client = AzureLLMClient(azure_config)
        print("✅ LLM Client initialized")
        
        # Test Planner
        planner = PredictivePlanner(azure_config)
        print("✅ Predictive Planner initialized")
        
        # Test Executor
        executor = EnhancedKGExecutor(azure_config)
        print("✅ KG Executor initialized")
        
        # Test Memory
        memory = ThreeLayerMemory(question)
        print("✅ Memory system initialized")
        
        # Simple plan test
        print("\n🧠 Testing planning...")
        
        plan_result = planner.decompose_plan_with_prediction(question, {})
        
        if plan_result.get("success", False):
            plan = plan_result.get("plan", [])
            print(f"✅ Plan created with {len(plan)} steps")
            for i, step in enumerate(plan, 1):
                print(f"   Step {i}: {step.get('description', 'No description')}")
        else:
            print(f"❌ Planning failed: {plan_result.get('error', 'Unknown error')}")
        
        print("\n🎉 Basic functionality test completed!")
        print("="*80)
        
        return {
            "success": True,
            "question": question,
            "components_tested": ["llm_client", "planner", "executor", "memory"],
            "plan_steps": len(plan_result.get("plan", [])) if plan_result.get("success") else 0
        }
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    main()
