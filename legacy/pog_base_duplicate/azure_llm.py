"""
Azure OpenAI LLM Interface for PPoGA
Enhanced with robust error handling and token tracking
"""

import openai
import json
import time
from typing import Dict, Any, Tuple, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_azure_config() -> Dict[str, Any]:
    """Get Azure OpenAI configuration from environment variables"""
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "api_base": os.getenv("OPENAI_API_BASE"),
        "api_type": os.getenv("OPENAI_API_TYPE"),
        "api_version": os.getenv("OPENAI_API_VERSION"),
        "deployment_id": os.getenv("DEPLOYMENT_ID"),
    }


def call_azure_openai(
    prompt: str,
    azure_config: Dict[str, Any],
    temperature: float = 0.3,
    max_tokens: int = 4096,
    print_in: bool = False,
    print_out: bool = False,
) -> Tuple[str, Dict[str, int]]:
    """
    Call Azure OpenAI API with robust error handling

    Args:
        prompt: Input prompt
        azure_config: Azure configuration dictionary
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        print_in: Whether to print input prompt
        print_out: Whether to print output

    Returns:
        Tuple of (response_text, token_usage)
    """
    if print_in:
        print(f"🔵 Input: {prompt[:200]}...")

    try:
        # Set up Azure OpenAI client
        client = openai.AzureOpenAI(
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            azure_endpoint=azure_config["api_base"],
        )

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {"role": "user", "content": prompt},
        ]

        # Call API
        completion = client.chat.completions.create(
            model=azure_config["deployment_id"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
        )

        result = completion.choices[0].message.content

        # Extract token usage
        token_usage = {
            "total": completion.usage.total_tokens,
            "input": completion.usage.prompt_tokens,
            "output": completion.usage.completion_tokens,
        }

        if print_out:
            print(f"🟢 Output: {result[:200]}...")

        return result, token_usage

    except Exception as e:
        print(f"❌ Azure OpenAI Error: {e}")
        # Return fallback response
        return "Error: Failed to get response from Azure OpenAI", {
            "total": 0,
            "input": 0,
            "output": 0,
        }


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response with robust parsing

    Args:
        response: Raw LLM response

    Returns:
        Parsed JSON dictionary
    """
    try:
        # Find JSON boundaries
        first_brace = response.find("{")
        last_brace = response.rfind("}")

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = response[first_brace : last_brace + 1]
            return json.loads(json_str)
        else:
            # Try to parse entire response
            return json.loads(response)

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing error: {e}")
        print(f"Response: {response[:200]}...")

        # Return fallback structure
        return {
            "error": "JSON parsing failed",
            "raw_response": response,
            "fallback": True,
        }


def extract_list_from_response(response: str) -> list:
    """
    Extract list from LLM response (for relation/entity lists)

    Args:
        response: Raw LLM response

    Returns:
        Parsed list
    """
    try:
        # Find list boundaries
        first_bracket = response.find("[")
        last_bracket = response.rfind("]")

        if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
            list_str = response[first_bracket : last_bracket + 1]
            return eval(list_str)
        else:
            # Try to parse entire response
            return eval(response.strip())

    except Exception as e:
        print(f"⚠️ List parsing error: {e}")
        print(f"Response: {response[:200]}...")

        # Return empty list as fallback
        return []


class TokenTracker:
    """Track token usage across API calls"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_calls = 0
        self.total_cost = 0.0  # Estimated cost in USD

    def add_usage(self, usage: Dict[str, int]):
        """Add token usage from API response"""
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)
        self.total_calls += 1

        # Estimate cost (approximate rates for GPT-4)
        prompt_cost = usage.get("prompt_tokens", 0) * 0.00003  # $0.03 per 1K tokens
        completion_cost = (
            usage.get("completion_tokens", 0) * 0.00006
        )  # $0.06 per 1K tokens
        self.total_cost += prompt_cost + completion_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary"""
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
        }

    def print_summary(self):
        """Print usage summary"""
        summary = self.get_summary()
        print(f"🎯 API Usage Summary:")
        print(f"   Total Calls: {summary['total_calls']}")
        print(f"   Prompt Tokens: {summary['prompt_tokens']:,}")
        print(f"   Completion Tokens: {summary['completion_tokens']:,}")
        print(f"   Total Tokens: {summary['total_tokens']:,}")
        print(f"   Estimated Cost: ${summary['estimated_cost_usd']}")


class AzureLLMClient:
    """Azure OpenAI LLM Client for PPoGA"""

    def __init__(self, azure_config: Dict[str, str]):
        """Initialize with Azure configuration"""
        # Convert config format for compatibility
        legacy_config = {
            "api_key": azure_config.get("api_key", ""),
            "api_base": azure_config.get("endpoint", ""),
            "api_type": "azure",
            "api_version": azure_config.get("api_version", "2024-08-01-preview"),
            "deployment_id": azure_config.get("deployment_name", ""),
        }

        self.config = legacy_config
        self.token_tracker = TokenTracker()

        # Validate configuration
        required_keys = ["api_key", "api_base", "deployment_id"]
        for key in required_keys:
            if key not in legacy_config or not legacy_config[key]:
                print(f"⚠️  Warning: Azure config missing '{key}'")

    def call(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        print_in: bool = False,
        print_out: bool = False,
    ) -> Tuple[str, Dict[str, int]]:
        """Call Azure OpenAI API"""
        return call_azure_openai(
            prompt=prompt,
            azure_config=self.config,
            temperature=temperature,
            max_tokens=max_tokens,
            print_in=print_in,
            print_out=print_out,
        )

    def extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response"""
        return extract_json_from_response(response)

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get token usage summary"""
        return self.token_tracker.get_summary()


# Global token tracker
token_tracker = TokenTracker()

# Global token tracker instance
global_token_tracker = TokenTracker()
