"""
Simple Azure OpenAI API Test
Test if the provided Azure credentials work
"""

import os
import openai
from typing import Dict, Any


# Set up Azure OpenAI configuration from .env file
def load_config() -> Dict[str, str]:
    """Load Azure config from environment"""
    # Try to load from .env file
    try:
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    except FileNotFoundError:
        print("No .env file found")

    return {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION", ""),
    }


def test_azure_api(config: Dict[str, str]) -> Dict[str, Any]:
    """Test Azure OpenAI API connection"""
    print(f"🧪 Testing Azure OpenAI API...")
    print(f"   Endpoint: {config['endpoint']}")
    print(f"   Deployment: {config['deployment']}")
    print(f"   API Version: {config['api_version']}")

    try:
        # Initialize Azure OpenAI client
        client = openai.AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
        )

        # Simple test prompt
        test_prompt = "What is 2+2? Answer briefly."

        print(f"🔵 Sending test prompt: {test_prompt}")

        response = client.chat.completions.create(
            model=config["deployment"],
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=50,
            temperature=0.3,
        )

        answer = response.choices[0].message.content
        usage = response.usage

        print(f"✅ API Response: {answer}")
        print(f"📊 Token Usage: {usage}")

        return {
            "success": True,
            "response": answer,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }

    except Exception as e:
        print(f"❌ API Test Failed: {str(e)}")
        return {"success": False, "error": str(e)}


def main():
    """Main test function"""
    print("🚀 Azure OpenAI API Test")
    print("=" * 50)

    # Load configuration
    config = load_config()

    # Validate configuration
    missing_keys = [k for k, v in config.items() if not v]
    if missing_keys:
        print(f"❌ Missing configuration keys: {missing_keys}")
        print("Please check your .env file")
        return

    # Test API
    result = test_azure_api(config)

    print("\n" + "=" * 50)
    if result["success"]:
        print("🎉 Azure OpenAI API Test PASSED!")
        print("   Your credentials are working correctly.")
        print("   You can now run the full PPoGA system.")
    else:
        print("😞 Azure OpenAI API Test FAILED!")
        print("   Please check your credentials and try again.")
    print("=" * 50)


if __name__ == "__main__":
    main()
