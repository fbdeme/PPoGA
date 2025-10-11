# PPoGA: Predictive Plan-on-Graph with Action

A revolutionary knowledge graph question answering system that combines the proven reliability of Plan-on-Graph (PoG) with advanced predictive capabilities inspired by PreAct and dynamic replanning from PLAN-AND-ACT.

## 🎯 Overview

PPoGA represents a complete reimagining of knowledge graph reasoning, built on a solid foundation of the original PoG codebase while incorporating:

- **✅ Proven SPARQL Engine**: Direct integration with PoG's battle-tested Freebase query system
- **🔮 Predictive Reasoning**: PreAct-inspired outcome prediction before action execution
- **🧠 Enhanced Memory**: Three-layer memory architecture (Strategic, Execution, Knowledge)
- **🔄 Dynamic Replanning**: PLAN-AND-ACT inspired strategic plan modification
- **🎯 Self-Correction**: Two-level correction mechanism for tactical and strategic failures
- **🧪 TDD Infrastructure**: Complete Test-Driven Development framework with 32 test cases

## 🏗️ Architecture

### Core Components

```
src/ppoga_project/
├── PoG/                          # 🔥 Proven PoG Engine (from original PoG research)
│   ├── freebase_func.py          # SPARQL execution engine
│   ├── utils.py                  # LLM utilities
│   └── prompt_list.py            # Base prompts
├── ppoga/                        # 🚀 PPoGA Extensions
│   ├── predictive_planner.py     # Strategic planning + PreAct prediction
│   ├── enhanced_executor.py      # PoG integration layer
│   └── enhanced_memory.py        # 3-layer memory system
├── main_ppoga.py                 # 🎬 Main execution engine
└── config.py                     # ⚙️ Configuration system

tests/                            # 🧪 Complete TDD Infrastructure
├── conftest.py                   # Global fixtures and configuration
├── fixtures/
│   ├── mock_responses.py         # LLM/SPARQL response mocks
│   └── sample_questions.py       # Test data sets
├── unit/                         # Unit tests (planner, executor, memory)
├── integration/                  # Integration tests (end-to-end)
└── benchmark/                    # Performance benchmarks
```

### 🧹 Development Environment

**✅ Standardized Environment**

- **Virtual Environment**: Standard `.venv` (Poetry removed for simplicity)
- **Development Tools**: `./dev.sh` script for all common operations
- **Code Quality**: Integrated black, isort, flake8, mypy with CI/CD
- **Testing**: pytest with comprehensive configuration and 32 test cases

### Three-Layer Memory Architecture

1. **Strategic Layer**: Plans, rationale, and strategic decisions
2. **Execution Layer**: Prediction-Action-Observation-Thought cycles
3. **Knowledge Layer**: Discovered entities, relations, and reasoning paths

### Six-Step Workflow

1. **Strategic Plan Decomposition**: Break complex questions into executable steps
2. **Outcome Prediction**: Predict results before taking action (PreAct)
3. **SPARQL Execution**: Use PoG's proven query engine for actual KG exploration
4. **Observation & Thought**: Compare predictions with reality and analyze
5. **Memory Update**: Update all three memory layers with new information
6. **Evaluation & Self-Correction**: Decide next action (proceed/correct/replan/finish)

## 🚀 Quick Start

### Prerequisites

1. **Freebase Setup**: Deploy local Freebase following [PoG instructions](https://github.com/liyichen-cly/PoG)
2. **Python Environment**: Python 3.11+ with required dependencies
3. **OpenAI API Key**: For LLM-powered planning and reasoning

### Installation

```bash
# Clone and setup
git clone <your-repo>
cd ppoga-project

# Setup environment (automated)
./dev.sh test  # Auto-creates .venv, installs deps, runs tests

# Manual setup alternative
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and Freebase endpoint
```

### Basic Usage

```bash
# Single question
python src/ppoga_project/main_ppoga.py \
    --question "Who directed The Godfather?" \
    --max_iterations 8

# With custom settings
python src/ppoga_project/main_ppoga.py \
    --question "What is the capital of France?" \
    --model gpt-4 \
    --max_iterations 5 \
    --verbose

# Dataset processing
python src/ppoga_project/main_ppoga.py \
    --dataset cwq \
    --max_iterations 10
```

### Development Commands

```bash
# Testing
./dev.sh test              # Run all tests
./dev.sh test-unit         # Unit tests only
./dev.sh test-integration  # Integration tests
./dev.sh test-benchmark    # Performance benchmarks

# Code Quality
./dev.sh lint              # Run all linting checks
./dev.sh format            # Auto-format code
./dev.sh coverage          # Generate coverage report

# Utilities
./dev.sh clean             # Clean cache files
./dev.sh install           # Install in development mode
```

### Environment Configuration

```bash
# Required
OPENAI_API_KEY=your_key_here
SPARQL_ENDPOINT=http://localhost:8890/sparql

# Optional
OPENAI_MODEL=gpt-3.5-turbo
MAX_ITERATIONS=10
MAX_DEPTH=4
VERBOSE=true
```

## 📊 Key Features

### 🔮 Predictive Reasoning (PreAct Integration)

- Predicts outcomes before executing KG queries
- Compares predictions with actual results
- Uses prediction accuracy to improve future planning

### 🧠 Enhanced Memory System

```python
# Strategic Layer
memory.strategy = {
    "overall_plan": [...],      # Strategic steps
    "plan_rationale": "...",    # Why this plan
    "replan_count": 0          # Adaptation tracking
}

# Execution Layer
memory.execution = {
    "cycles": {...},           # Prediction-Action-Observation-Thought
    "current_prediction": {}, # What we expect
    "current_observation": "" # What actually happened
}

# Knowledge Layer
memory.knowledge = {
    "discovered_entities": {}, # Entity mappings
    "exploration_graph": {},   # KG structure discovered
    "reasoning_chains": []     # Logical reasoning paths
}
```

### 🔄 Dynamic Replanning (PLAN-AND-ACT Integration)

- **Level 1 (Path Correction)**: Retry same plan with different approach
- **Level 2 (Strategic Replanning)**: Abandon current plan, create new strategy

### ⚡ Proven SPARQL Integration

- Direct use of PoG's `freebase_func.py` for KG queries
- Inherited reliability and performance optimizations
- Seamless entity resolution and relation discovery

### 🧪 Test-Driven Development

- **32 test cases** across unit/integration/benchmark categories
- **Mock system** for LLM and SPARQL responses
- **CI/CD pipeline** with GitHub Actions
- **Coverage reporting** with comprehensive metrics

## 📈 Performance Benefits

Compared to previous implementations:

- **✅ Real KG Queries**: No more mock executors - actual SPARQL execution
- **🎯 Strategic Planning**: Coherent multi-step reasoning instead of ad-hoc exploration
- **🧠 Memory Persistence**: Information retention across planning cycles
- **🔮 Predictive Accuracy**: Learn from prediction vs reality comparisons
- **🔄 Adaptive Planning**: Dynamic strategy modification based on results
- **🧪 Quality Assurance**: Comprehensive testing infrastructure ensures reliability

## 🔧 Advanced Configuration

### Custom LLM Settings

```python
config = PPoGAConfig(
    model="gpt-4",
    temperature_exploration=0.1,  # More focused exploration
    temperature_reasoning=0.3,    # Balanced reasoning
    max_length=8192              # Longer responses
)
```

### SPARQL Endpoint Configuration

```python
config = PPoGAConfig(
    sparql_endpoint="http://your-freebase:8890/sparql",
    remove_unnecessary_rel=True  # Filter noise relations
)
```

### Execution Control

```python
config = PPoGAConfig(
    max_iterations=15,            # Longer exploration
    max_depth=6,                 # Deeper KG traversal
    prediction_confidence_threshold=0.8  # Higher accuracy bar
)
```

## 📚 Research Background

PPoGA synthesizes insights from three foundational papers:

1. **Plan-on-Graph (PoG)** - Self-correcting adaptive planning with backtracking
2. **PreAct** - Predictive reasoning through outcome prediction
3. **PLAN-AND-ACT** - Dynamic replanning with role separation (Planner/Executor)

This implementation provides the first working system that meaningfully combines all three methodologies with proven KG infrastructure.

## ✅ Validation & Testing

### 🧪 System Verification

- **Execution Test**: Successfully runs with proper argument handling
- **Import Resolution**: All module dependencies correctly resolved
- **Code Quality**: Clean architecture with comprehensive TDD infrastructure

### 📊 Architecture Validation

- **✅ PreAct Integration**: Prediction → Action → Observation → Thought cycles implemented
- **✅ PoG Foundation**: Direct integration with proven SPARQL engine (battle-tested)
- **✅ Memory System**: 3-layer architecture (Strategic/Execution/Knowledge) operational
- **✅ Self-Correction**: Multi-level error handling and replanning mechanisms
- **✅ TDD Infrastructure**: 32 test cases with unit/integration/benchmark coverage

### 🗂️ Code Organization (October 2025)

**Systematic v2 → Official Promotion:**

- **Directory Rename**: `ppoga_v2/` → `ppoga/`
- **Main Script**: `main_ppoga_v2.py` → `main_ppoga.py`
- **Function Updates**: `run_ppoga_v2` → `run_ppoga`
- **Import Fixes**: Resolved all relative import issues in PoG module
- **Documentation**: Updated all references from "PPoGA v2" to "PPoGA"

**Development Infrastructure:**

- **Environment**: Standardized on `.venv` virtual environment
- **Testing**: Complete pytest framework with 32 test cases
- **CI/CD**: GitHub Actions with multi-Python version testing
- **Code Quality**: Integrated black, isort, flake8, mypy tooling

## 🚀 Recent Updates

### Official Release - PPoGA (October 11, 2025)

- 🎯 **v2 → Official**: Systematic promotion to official PPoGA system
- 🧪 **TDD Complete**: 32 test cases across unit/integration/benchmark
- 🛠️ **Development Tools**: `./dev.sh` script with all common operations
- 🔧 **Environment**: Standardized `.venv` setup replacing Poetry
- 📝 **Documentation**: Complete C4 architecture documentation
- ✅ **Import Fixes**: Resolved all module dependency issues
- � **Main Script**: Updated to `main_ppoga.py` as primary entry point

### Core Capabilities

- 🔥 **PoG Integration**: Built on proven NeurIPS 2024 research foundation
- 🔮 **Predictive Planning**: PreAct-inspired outcome prediction system
- 🧠 **Enhanced Memory**: Three-layer memory architecture implementation
- 🎯 **Self-Correction**: Dynamic replanning and error correction mechanisms
- 🧪 **Quality Assurance**: Comprehensive testing and CI/CD infrastructure

## 🤝 Contributing

This project bridges research and implementation. Contributions welcome in:

- Enhanced entity resolution strategies
- Advanced relation selection algorithms
- Improved prediction accuracy metrics
- Additional dataset integrations (WebQSP, GrailQA)
- Performance optimizations
- Test coverage expansion

### Development Workflow

```bash
# Setup development environment
./dev.sh test

# Make changes and test
./dev.sh test-unit
./dev.sh lint
./dev.sh format

# Run full test suite
./dev.sh test
./dev.sh coverage
```

## 📄 License

This project builds upon the Apache 2.0 licensed PoG codebase and maintains the same license for new contributions.

## 🙏 Acknowledgments

- **PoG Team**: For providing the robust SPARQL foundation
- **PreAct Authors**: For predictive reasoning insights
- **PLAN-AND-ACT Authors**: For dynamic planning methodologies
