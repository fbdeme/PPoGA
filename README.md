# PPoGA: Predictive Plan-on-Graph with Action

A sophisticated knowledge graph question answering system that combines the strengths of Plan-on-Graph, PLAN-AND-ACT, and PreAct methodologies, implemented with Azure OpenAI integration and Poetry dependency management.

## 🎯 Overview

PPoGA (Predictive Plan-on-Graph with Action) is an advanced AI agent framework designed for complex knowledge graph question answering. It features:

- **Strategic Planning**: Decomposes complex questions into executable sub-goals
- **Predictive Execution**: Predicts outcomes before taking actions (inspired by PreAct)
- **Role Separation**: Clear distinction between Planner (strategic) and Executor (operational) components
- **Self-Correction**: Two-level correction mechanism (tactical and strategic)
- **Robust Implementation**: Enhanced error handling and JSON parsing capabilities

## 🏗️ Architecture

### Core Components

1. **Planner (Strategic Brain)**: 
   - Plan decomposition and strategy formulation
   - Outcome prediction and evaluation
   - Self-correction and replanning capabilities

2. **Executor (Operational Engine)**:
   - Knowledge graph query execution
   - Result processing and summarization
   - Entity and relation discovery

3. **Memory System**:
   - Strategic layer: Plans, rationale, and status
   - Execution layer: Predictions, observations, and logs
   - Knowledge layer: Discovered entities and reasoning paths

### Implementation Approaches

This project provides two implementation approaches:

1. **Robust Azure Implementation** (`main_robust_azure_ppoga.py`):
   - Direct workflow control with enhanced error handling
   - Intelligent fallback mechanisms for JSON parsing failures
   - Comprehensive logging and statistics tracking

2. **LangGraph Implementation** (`langgraph_ppoga.py`):
   - State-based workflow modeling using LangGraph
   - Clear visual representation of the agent workflow
   - Modular node-based architecture for easy maintenance

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Azure OpenAI API access

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ppoga-project
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

3. Install dependencies:
```bash
poetry install
```

### Configuration

Set up your Azure OpenAI credentials by modifying the environment variables in the main scripts:

```python
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_API_BASE"] = "your-azure-endpoint"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2024-05-01-preview"
os.environ["DEPLOYMENT_ID"] = "your-deployment-id"
```

### Usage

#### Robust Azure Implementation

```bash
poetry run python src/ppoga_project/main_robust_azure_ppoga.py \
    --question "Which of Taylor Swift's songs has won American Music Awards?" \
    --max_iterations 6 \
    --output_file results.json
```

#### LangGraph Implementation

```bash
poetry run python src/ppoga_project/langgraph_ppoga.py \
    --question "Which of Taylor Swift's songs has won American Music Awards?" \
    --max_iterations 5 \
    --output_file langgraph_results.json
```

## 📊 Example Results

### Sample Question: "Which of Taylor Swift's songs has won American Music Awards?"

**Robust Azure Implementation Result:**
```json
{
  "success": true,
  "question": "Which of Taylor Swift's songs has won American Music Awards?",
  "answer": "Taylor Swift's song 'Blank Space' has won an American Music Award.",
  "confidence": "high",
  "execution_time": 9.77,
  "iterations": 1,
  "statistics": {
    "llm_calls": 5,
    "total_tokens_used": 3083,
    "kg_queries": 1,
    "entities_discovered": 16
  }
}
```

**LangGraph Implementation Result:**
```json
{
  "success": true,
  "question": "Which of Taylor Swift's songs has won American Music Awards?",
  "answer": "'Anti-Hero' by Taylor Swift has won American Music Awards.",
  "confidence": "high",
  "execution_time": 9.97,
  "iterations": 1,
  "statistics": {
    "llm_calls": 5,
    "kg_queries": 1,
    "entities_discovered": 16
  },
  "workflow_type": "LangGraph StateGraph"
}
```

## 🔧 Key Features

### Enhanced Prompts
All prompts are written in English following academic standards and based on the original research papers:
- Plan decomposition prompts inspired by PLAN-AND-ACT
- Prediction prompts based on PreAct methodology
- Evaluation and reasoning prompts enhanced from Plan-on-Graph

### Robust JSON Parsing
Multiple parsing strategies to handle various LLM response formats:
- Direct JSON parsing
- Markdown code block extraction
- Pattern-based key-value extraction
- Intelligent fallback mechanisms

### Comprehensive Error Handling
- Graceful degradation when parsing fails
- Intelligent fallback responses based on context
- Detailed logging and error reporting
- Automatic retry mechanisms with path correction

### Performance Monitoring
- Token usage tracking
- Execution time measurement
- Success rate monitoring
- Detailed execution logs

## 📁 Project Structure

```
ppoga-project/
├── src/ppoga_project/
│   ├── ppoga/
│   │   ├── enhanced_prompts.py          # Academic-quality English prompts
│   │   ├── azure_planner.py             # Basic Azure OpenAI planner
│   │   └── robust_azure_planner.py      # Enhanced planner with error handling
│   ├── main_robust_azure_ppoga.py       # Robust implementation main script
│   └── langgraph_ppoga.py               # LangGraph-based implementation
├── pyproject.toml                       # Poetry configuration
├── README.md                           # This file
└── results/                            # Output directory for results
```

## 🔬 Research Foundation

This implementation is based on three seminal papers:

1. **Plan-on-Graph**: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs
2. **PLAN-AND-ACT**: Improving Planning of Agents for Long-Horizon Tasks
3. **PreAct**: Prediction Enhances Agent's Planning Ability

The system combines:
- PoG's knowledge graph exploration and self-correction mechanisms
- PLAN-AND-ACT's role separation and dynamic replanning
- PreAct's predictive planning capabilities

## 🎛️ Configuration Options

### Azure OpenAI Settings
- `temperature`: Controls response randomness (default: 0.1 for consistency)
- `max_tokens`: Maximum tokens per response (default: 2048)
- `api_version`: Azure OpenAI API version

### Execution Parameters
- `max_iterations`: Maximum workflow iterations (default: 6-8)
- `max_consecutive_failures`: Failure tolerance before forcing progression
- `replan_limit`: Maximum number of replanning attempts

## 📈 Performance Characteristics

### Typical Performance Metrics
- **Execution Time**: 8-30 seconds depending on question complexity
- **Token Usage**: 3,000-10,000 tokens per question
- **Success Rate**: High success rate with robust error handling
- **Scalability**: Handles complex multi-step reasoning tasks

### Comparison: Robust vs LangGraph Implementation

| Aspect | Robust Implementation | LangGraph Implementation |
|--------|----------------------|-------------------------|
| **Complexity** | Direct workflow control | State-based modeling |
| **Maintainability** | Good | Excellent |
| **Performance** | Slightly faster | Comparable |
| **Debugging** | Manual logging | Built-in state tracking |
| **Extensibility** | Moderate | High |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Plan-on-Graph research team
- PLAN-AND-ACT methodology authors
- PreAct framework developers
- LangGraph and LangChain communities
- Azure OpenAI team for API access

---

**Note**: This implementation uses mock knowledge graph data for demonstration purposes. In a production environment, you would integrate with actual knowledge graph databases like Freebase, Wikidata, or custom knowledge graphs.
