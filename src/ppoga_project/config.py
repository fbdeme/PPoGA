import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PPoGAConfig:
    """Configuration for PPoGA v2 system"""

    # Basic settings
    question: str = ""
    max_iterations: int = 10
    max_depth: int = 4

    # LLM settings
    openai_api_key: str = ""
    model: str = "gpt-3.5-turbo"
    temperature_exploration: float = 0.3
    temperature_reasoning: float = 0.3
    max_length: int = 4096

    # SPARQL/Freebase settings
    sparql_endpoint: str = "http://localhost:8890/sparql"
    remove_unnecessary_rel: bool = True

    # PPoGA specific settings
    enable_prediction: bool = True
    enable_memory_system: bool = True
    prediction_confidence_threshold: float = 0.7

    # Dataset settings
    dataset: str = "cwq"  # cwq, webqsp, grailqa
    cope_alias_dir: str = "cope_alias"  # Directory for alias files
    batch_size: int = 1  # Process questions in batches

    # Output settings
    output_dir: str = "results"
    save_intermediate_results: bool = True
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "PPoGAConfig":
        """Create configuration from environment variables"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            sparql_endpoint=os.getenv(
                "SPARQL_ENDPOINT", "http://localhost:8890/sparql"
            ),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            max_depth=int(os.getenv("MAX_DEPTH", "4")),
            dataset=os.getenv("DATASET", "cwq"),
            output_dir=os.getenv("OUTPUT_DIR", "results"),
            verbose=bool(os.getenv("VERBOSE", "True").lower() in ["true", "1", "yes"]),
        )

    def to_args_namespace(self):
        """Convert to args namespace for compatibility with PoG functions"""

        class Args:
            def __init__(self, config):
                self.dataset = config.dataset
                self.max_length = config.max_length
                self.temperature_exploration = config.temperature_exploration
                self.temperature_reasoning = config.temperature_reasoning
                self.depth = config.max_depth
                self.remove_unnecessary_rel = config.remove_unnecessary_rel
                self.LLM_type = config.model
                self.openai_api_keys = config.openai_api_key

        return Args(self)

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dictionary"""
        return {
            "api_key": self.openai_api_key,
            "model": self.model,
            "temperature_exploration": self.temperature_exploration,
            "temperature_reasoning": self.temperature_reasoning,
            "max_length": self.max_length,
        }

    def get_kg_config(self) -> Dict[str, Any]:
        """Get knowledge graph configuration dictionary"""
        return {
            "sparql_endpoint": self.sparql_endpoint,
            "remove_unnecessary_rel": self.remove_unnecessary_rel,
        }

    def validate(self) -> bool:
        """Validate configuration"""
        if not self.openai_api_key:
            print("❌ OpenAI API key is required")
            return False

        if not self.question and not self.dataset:
            print("❌ Either question or dataset must be specified")
            return False

        return True
