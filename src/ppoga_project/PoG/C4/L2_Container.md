# C4 Level 2 - Container Diagram (Original PoG)

## Container Overview

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                Original PoG System                                 │
│                                                                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐               │
│  │                 │    │                 │    │                 │               │
│  │  Main Runner    │───▶│  Core Reasoning │───▶│   Memory &      │               │
│  │   Container     │    │   Container     │    │  Output System  │               │
│  │                 │    │                 │    │                 │               │
│  │ main_freebase.py│    │ freebase_func.py│    │   File System   │               │
│  │                 │    │    utils.py     │    │   + JSONL       │               │
│  │ Python Script   │    │ prompt_list.py  │    │   Storage       │               │
│  └─────────────────┘    │                 │    │                 │               │
│                          │ Python Package  │    │ File System     │               │
│                          └─────────────────┘    └─────────────────┘               │
└────────────────────────────────────────────────────────────────────────────────────┘

External Dependencies:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenAI API    │    │     Freebase    │    │   Sentence      │    │    Dataset      │
│    Service      │    │ SPARQL Endpoint │    │  Transformers   │    │     Files       │
│                 │    │                 │    │                 │    │                 │
│   REST/HTTPS    │    │   HTTP/SPARQL   │    │  Python Library │    │   JSON/JSONL    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Container Details

### 1. Main Runner Container

**File**: `main_freebase.py`  
**Technology**: Python Script  
**Purpose**: Orchestration and execution control

**Key Responsibilities**:

- Parse command-line arguments and configuration
- Load and process datasets (CWQ, WebQSP, GrailQA)
- Manage execution flow for each question
- Handle dataset iteration and progress tracking
- Coordinate between reasoning and memory systems
- Error handling and recovery

**Configuration Parameters**:

```python
--dataset          # Dataset to process (cwq, webqsp, grailqa)
--max_length       # Maximum LLM response length (4096)
--temperature_exploration  # LLM temperature for exploration (0.3)
--temperature_reasoning    # LLM temperature for reasoning (0.3)
--depth            # Maximum exploration depth (4)
--remove_unnecessary_rel   # Enable relation filtering (True)
--LLM_type         # Language model to use (gpt-3.5-turbo)
--opeani_api_keys  # OpenAI API credentials
```

**Dependencies**:

- Core Reasoning Container
- Memory & Output System
- External dataset files
- Configuration management

---

### 2. Core Reasoning Container

**Files**: `freebase_func.py`, `utils.py`, `prompt_list.py`  
**Technology**: Python Package  
**Purpose**: Knowledge graph reasoning and optimization

#### Sub-modules:

**A. SPARQL Engine (`freebase_func.py`)**

- **19 specialized functions** for knowledge graph operations
- SPARQL query execution and optimization
- Entity and relation management
- Intelligent pruning algorithms

**B. LLM Interface (`utils.py`)**

- **14 utility functions** for language model interaction
- Response parsing and validation
- Question decomposition
- Result serialization

**C. Prompt Library (`prompt_list.py`)**

- **7 specialized prompts** for different reasoning phases
- Battle-tested prompt templates
- Task-specific prompt formatting

**Key Optimizations**:

- Early stopping logic (`half_stop`)
- Intelligent entity pruning (`entity_condition_prune`)
- Memory-guided exploration (`update_memory`)
- Similarity-based ranking (`retrieve_top_docs`)

**Dependencies**:

- Freebase SPARQL endpoint
- OpenAI API services
- Sentence Transformers library
- Memory & Output System

---

### 3. Memory & Output System

**Technology**: File System + JSONL Storage  
**Purpose**: State management and result persistence

**Memory Management**:

- File-based memory storage in `../mem/` directory
- Question-specific memory directories
- Memory state updates during exploration
- Context preservation across reasoning steps

**Output Management**:

- JSONL format for batch processing
- Structured result format with metadata
- Progress tracking and partial results
- Error state preservation

**File Structure**:

```
../mem/
├── {dataset}/
│   ├── {LLM_type}/
│   │   ├── {question_prefix}/
│   │   │   ├── mem          # Memory state file
│   │   │   └── subq         # Sub-questions file
results/
├── PoG_{dataset}_{model}.jsonl  # Final results
```

**Dependencies**:

- Local file system
- Core Reasoning Container

## Data Flow Architecture

### Question Processing Pipeline

```
Input Question
      │
      ▼
┌─────────────────┐
│ Question        │
│ Decomposition   │
│ (break_question)│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Topic Entity    │
│ Initialization  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Depth-based     │────▶│ Relation Search │────▶│ Entity Search   │
│ Exploration     │     │ & Pruning       │     │ & Filtering     │
│ Loop            │     │                 │     │                 │
└─────────┬───────┘     └─────────────────┘     └─────────────────┘
          │                                                ▲
          │                                                │
          ▼                                                │
┌─────────────────┐     ┌─────────────────┐              │
│ Memory Update   │────▶│ Answer          │──────────────┘
│ & Evaluation    │     │ Generation      │
└─────────┬───────┘     └─────────────────┘
          │
          ▼
┌─────────────────┐
│ Early Stop      │
│ Decision        │
└─────────┬───────┘
          │
          ▼
     Final Answer
```

### Memory State Flow

```
Question Input
      │
      ▼
┌─────────────────┐
│ Initialize      │
│ Memory Directory│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐
│ Update Memory   │◄────┤ Exploration     │
│ with Discoveries│     │ Results         │
└─────────┬───────┘     └─────────────────┘
          │
          ▼
┌─────────────────┐     ┌─────────────────┐
│ Provide Context │────▶│ Next LLM Call   │
│ for LLM         │     │                 │
└─────────────────┘     └─────────────────┘
```

## Technology Stack

### Core Runtime

- **Language**: Python 3.8+
- **Architecture**: Functional programming with file-based state

### Key Dependencies

```python
# SPARQL and Knowledge Graph
SPARQLWrapper==1.8.5        # SPARQL query execution
requests                     # HTTP communication

# Language Models and NLP
openai                       # OpenAI API integration
sentence-transformers        # Semantic similarity

# Data Processing
json                         # Result serialization
re                          # Response parsing
time                        # Performance tracking
random                      # Sampling and selection

# Scientific Computing
numpy                       # Numerical operations (via sentence-transformers)
```

### External Service Interfaces

**OpenAI API**:

```python
def run_llm(prompt, temperature, max_tokens, api_keys, engine):
    # Handles API calls with error management
    # Supports multiple models (GPT-3.5, GPT-4)
    # Returns response and token usage statistics
```

**SPARQL Endpoint**:

```python
def execurte_sparql(sparql_query):
    # Connects to localhost:8890/sparql
    # Executes SPARQL queries against Freebase
    # Returns structured results in JSON format
```

**File System**:

```python
def save_2_jsonl(question, answer, metadata, filename):
    # Appends results to JSONL files
    # Maintains structured output format
    # Includes execution statistics and traces
```

## Deployment Architecture

### Local Development Setup

```bash
# Required services
virtuoso-opensource         # Local Freebase SPARQL endpoint
python>=3.8                  # Runtime environment

# Configuration
export OPENAI_API_KEY="sk-..."
./virtuoso start            # Start SPARQL endpoint on :8890
python main_freebase.py --dataset cwq --LLM_type gpt-3.5-turbo
```

### Production Considerations

**Scalability**:

- Stateless question processing (except file-based memory)
- Horizontal scaling through dataset partitioning
- Memory cleanup between questions

**Reliability**:

- API retry logic with exponential backoff
- Graceful degradation when external services fail
- Checkpoint-based recovery for long-running datasets

**Performance**:

- Optimized SPARQL query patterns
- LLM response caching (implicitly through memory)
- Early stopping to prevent unnecessary computation

## Container Communication Patterns

### Synchronous Processing

- Main Runner orchestrates sequential question processing
- Core Reasoning performs blocking operations
- Memory System provides immediate state persistence

### Error Handling Strategy

```python
try:
    # Question processing pipeline
except SPARQLException:
    # SPARQL endpoint failure - skip question
except OpenAIException:
    # API failure - retry with backoff
except MemoryException:
    # File system issue - continue without memory
```

### Resource Management

- Memory directories cleaned between questions
- API rate limiting through built-in delays
- Configurable depth limits to prevent infinite exploration
