"""
Global pytest configuration and fixtures for PPoGA tests
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
import pytest

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root directory path"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_path(project_root_path):
    """Return the test data directory path"""
    return project_root_path / "tests" / "fixtures"


@pytest.fixture(scope="session")
def llm_config():
    """Mock LLM configuration for testing"""
    return {
        "api_key": "test_api_key",
        "model": "gpt-3.5-turbo",
        "temperature_exploration": 0.3,
        "temperature_reasoning": 0.3,
        "max_length": 4096,
        "endpoint": "https://test.openai.azure.com/",
        "api_version": "2023-05-15",
    }


@pytest.fixture(scope="function")
def mock_llm_response():
    """Mock LLM response for testing"""
    return {
        "choices": [{"message": {"content": "Test LLM response", "role": "assistant"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture(scope="function")
def sample_question():
    """Sample question for testing"""
    return "Who directed The Godfather?"


@pytest.fixture(scope="function")
def complex_question():
    """Complex multi-hop question for testing"""
    return "Who is the spouse of the director of The Godfather?"


@pytest.fixture(scope="function")
def sample_entities():
    """Sample entities for testing"""
    return {
        "The Godfather": "m.047z4b5",
        "Francis Ford Coppola": "m.0c4dn",
        "Eleanor Coppola": "m.0c4dp",
    }


@pytest.fixture(scope="function")
def sample_relations():
    """Sample relations for testing"""
    return ["film.film.directed_by", "people.person.spouse_s", "people.marriage.spouse"]


@pytest.fixture(scope="function")
def sample_triplets():
    """Sample knowledge graph triplets"""
    return [
        ["The Godfather", "film.film.directed_by", "Francis Ford Coppola"],
        ["Francis Ford Coppola", "people.person.spouse_s", "m.marriage1"],
        ["m.marriage1", "people.marriage.spouse", "Eleanor Coppola"],
    ]


@pytest.fixture(scope="function")
def memory_test_data():
    """Test data for memory system"""
    return {
        "question": "Who directed The Godfather?",
        "initial_entities": {"The Godfather": "m.047z4b5"},
        "discovered_entities": {
            "Francis Ford Coppola": "m.0c4dn",
            "The Godfather": "m.047z4b5",
        },
        "explored_relations": {
            "m.047z4b5": ["film.film.directed_by", "film.film.starring"]
        },
        "reasoning_chains": [
            [["The Godfather", "film.film.directed_by", "Francis Ford Coppola"]]
        ],
    }


# Pytest hooks for better test output
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "mock: mark test as using mocking")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unit marker to unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add benchmark marker to benchmark tests
        elif "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)


def pytest_runtest_setup(item):
    """Setup for each test"""
    # Skip slow tests by default unless --slow flag is provided
    if "slow" in item.keywords and not item.config.getoption("--slow", default=False):
        pytest.skip("Skipping slow test (use --slow to run)")


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--benchmark", action="store_true", default=False, help="run benchmark tests"
    )
