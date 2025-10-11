# C4 Level 2 - Container Diagram

## Container Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                PPoGA System                                      │
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │                 │    │                 │    │                 │             │
│  │   Main Runner   │    │   PPoGA Core    │    │   PoG Legacy    │             │
│  │   Container     │◄───┤   Container     │◄───┤   Container     │             │
│  │                 │    │                 │    │                 │             │
│  │ Python Script   │    │ Python Package  │    │ Python Package  │             │
│  └─────────────────┘    └─────────┬───────┘    └─────────────────┘             │
│                                   │                                             │
│                          ┌─────────▼───────┐                                   │
│                          │                 │                                   │
│                          │   Evaluation    │                                   │
│                          │   Framework     │                                   │
│                          │   Container     │                                   │
│                          │                 │                                   │
│                          │ Python Package  │                                   │
│                          └─────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

External Dependencies:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenAI/Azure  │    │     Freebase    │    │    Dataset      │
│   API Service   │    │ SPARQL Endpoint │    │     Files       │
│                 │    │                 │    │                 │
│   REST/HTTP     │    │   HTTP/SPARQL   │    │   JSON/JSONL    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Container Details

### 1. Main Runner Container

**Technology**: Python Script  
**Purpose**: Entry point and orchestration layer

**Key Files**:

- `main_ppoga.py` - Primary execution script
- `config.py` - Configuration management

**Responsibilities**:

- Parse command-line arguments
- Initialize system components
- Coordinate execution flow between containers
- Handle error management and logging

**Dependencies**:

- PPoGA Core Container
- Configuration files
- Dataset files

---

### 2. PPoGA Core Container

**Technology**: Python Package  
**Purpose**: Advanced predictive planning and memory management

**Key Modules**:

- `enhanced_memory.py` - 3-layer memory architecture
- `predictive_planner.py` - Strategic planning with LLM prediction
- `enhanced_executor.py` - Execution coordination

**Responsibilities**:

- Decompose questions into strategic plans
- Predict optimal exploration paths
- Manage 3-layer memory system (Strategy/Execution/Knowledge)
- Coordinate with PoG Legacy for actual KG operations
- Log detailed execution traces

**Key Innovations**:

- Prediction-Action-Observation-Thought (PAOT) cycles
- Dynamic replanning capabilities
- Rich execution metadata and statistics

**Dependencies**:

- PoG Legacy Container (for SPARQL operations)
- LLM API services
- Configuration management

---

### 3. PoG Legacy Container

**Technology**: Python Package  
**Purpose**: Proven SPARQL optimization and knowledge graph operations

**Key Modules**:

- `freebase_func.py` - SPARQL query execution and optimization
- `utils.py` - LLM utilities and response parsing
- `prompt_list.py` - Proven prompt templates

**Responsibilities**:

- Execute optimized SPARQL queries
- Perform relation and entity search with pruning
- Handle knowledge graph data processing
- Provide battle-tested optimization functions

**Key Optimizations**:

- Relation pruning with LLM guidance
- Entity search optimization
- Response parsing and error handling

**Dependencies**:

- Freebase SPARQL endpoint
- LLM API services

---

### 4. Evaluation Framework Container

**Technology**: Python Package  
**Purpose**: Performance measurement and compatibility with original PoG

**Key Modules**:

- `eval.py` - Evaluation orchestration
- `utils.py` - Evaluation utilities and metrics
- `convert_results.py` - Format conversion utilities

**Responsibilities**:

- Load and process benchmark datasets (CWQ, WebQSP, GrailQA)
- Calculate evaluation metrics (Exact Match, F1)
- Convert between JSON and JSONL result formats
- Provide compatibility with original PoG evaluation

**Supported Datasets**:

- ComplexWebQuestions (CWQ)
- WebQuestionsSP (WebQSP)
- GrailQA

**Dependencies**:

- Dataset files
- Result files from both systems

## Data Flow

1. **Input**: Researcher provides question via Main Runner
2. **Planning**: Main Runner delegates to PPoGA Core for strategic decomposition
3. **Execution**: PPoGA Core coordinates with PoG Legacy for KG operations
4. **Memory**: PPoGA Core maintains state and learning across steps
5. **Evaluation**: Results flow to Evaluation Framework for performance measurement
6. **Output**: Final answer returned to researcher

## Technology Stack

**Core Runtime**: Python 3.8+
**Dependencies**:

- `SPARQLWrapper` - SPARQL query execution
- `openai` - LLM API integration
- `json`, `time`, `dataclasses` - Built-in Python modules

**External Services**:

- OpenAI/Azure API - Language model services
- Local Freebase endpoint - Knowledge graph access

## Deployment Considerations

**Local Development**:

- Requires local Freebase SPARQL endpoint (localhost:8890)
- API keys for LLM services
- Python environment with required dependencies

**Scalability**:

- Stateless design allows horizontal scaling
- Memory management confined to individual question processing
- External service dependencies may become bottlenecks
