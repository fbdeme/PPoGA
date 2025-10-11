# C4 Level 3 - Component Diagrams

## PPoGA Core Container Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PPoGA Core Container                                 │
│                                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐       │
│  │                  │     │                  │     │                  │       │
│  │  PredictivePlanner│────▶│ EnhancedExecutor │────▶│  PPoGAMemory     │       │
│  │                  │     │                  │     │                  │       │
│  │ - decompose_plan │     │ - execute_step   │     │ - strategy       │       │
│  │ - predict_outcome│     │ - coordinate_kg  │     │ - execution      │       │
│  │ - think_evaluate │     │ - manage_results │     │ - knowledge      │       │
│  │ - observe        │     │                  │     │ - statistics     │       │
│  └──────────┬───────┘     └─────────┬────────┘     └──────────────────┘       │
│             │                       │                                         │
│             │                       │                                         │
│             ▼                       ▼                                         │
│  ┌──────────────────┐     ┌──────────────────┐                               │
│  │                  │     │                  │                               │
│  │   LLM Interface  │     │   PoG Interface  │                               │
│  │                  │     │                  │                               │
│  │ - api_call       │     │ - relation_search│                               │
│  │ - response_parse │     │ - entity_search  │                               │
│  │ - error_handling │     │ - provide_triple │                               │
│  └──────────────────┘     └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. PredictivePlanner

**Purpose**: Strategic question decomposition and predictive reasoning

**Key Methods**:

- `decompose_plan(question, context)` - Break question into strategic steps
- `predict_step_outcome(step, context)` - Predict execution results
- `think_and_evaluate(prediction, observation)` - Analyze and learn
- `observe(action_result, context)` - Process execution feedback

**Unique Features**:

- 11 specialized prompts for different reasoning phases
- Context-aware planning based on current knowledge state
- Adaptive planning with replanning capabilities

#### 2. EnhancedExecutor

**Purpose**: Bridge between strategic planning and knowledge graph operations

**Key Methods**:

- `execute_step(entity_id, entity_name, plan_step)` - Execute individual plan steps
- `coordinate_kg_operations()` - Manage SPARQL queries via PoG interface
- `manage_results()` - Process and structure execution results

**Integration Points**:

- Uses PoG Legacy functions for actual KG operations
- Translates strategic plans into concrete actions
- Manages execution statistics and error handling

#### 3. PPoGAMemory

**Purpose**: 3-layer memory architecture for state management

**Memory Layers**:

- **Strategic Layer**: Plans, rationale, and strategic decisions
- **Execution Layer**: PAOT cycles and action history
- **Knowledge Layer**: Discovered entities, relations, reasoning chains

**Key Methods**:

- `update_plan()` - Update strategic planning state
- `add_execution_cycle()` - Record PAOT cycle
- `add_discovered_entities()` - Update knowledge layer
- `get_context_for_llm()` - Provide formatted context for LLM calls

#### 4. LLM Interface

**Purpose**: Abstraction layer for language model interactions

**Responsibilities**:

- API call management with retry logic
- Response parsing and validation
- Error handling and fallback strategies
- Token usage tracking

#### 5. PoG Interface

**Purpose**: Integration layer with PoG Legacy container

**Imported Functions**:

- `relation_search_prune()` - Optimized relation discovery
- `entity_search()` - Entity retrieval from KG
- `provide_triple()` - Triple formatting and processing

---

## PoG Legacy Container Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PoG Legacy Container                                 │
│                                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐       │
│  │                  │     │                  │     │                  │       │
│  │  SPARQL Engine   │────▶│  Optimization    │────▶│  Response Parser │       │
│  │                  │     │     Layer        │     │                  │       │
│  │ - execute_sparql │     │ - relation_prune │     │ - extract_reason │       │
│  │ - abandon_rels   │     │ - entity_prune   │     │ - extract_answer │       │
│  │ - replace_prefix │     │ - select_relations│     │ - parse_response │       │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘       │
│                                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐                               │
│  │                  │     │                  │                               │
│  │   Prompt Library │     │   Utils Library  │                               │
│  │                  │     │                  │                               │
│  │ - extract_relation│     │ - run_llm       │                               │
│  │ - answer_prompt  │     │ - break_question │                               │
│  │ - cot_prompt     │     │ - save_results   │                               │
│  └──────────────────┘     └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. SPARQL Engine

**Purpose**: Core knowledge graph query execution

**Key Functions**:

- `execute_sparql()` - Execute SPARQL queries against Freebase
- `abandon_rels()` - Filter unnecessary relations
- `replace_*_prefix()` - Clean and normalize URIs

#### 2. Optimization Layer

**Purpose**: Query optimization and intelligent pruning

**Key Functions**:

- `relation_search_prune()` - LLM-guided relation selection
- `entity_search()` - Optimized entity retrieval
- `provide_triple()` - Triple processing and formatting

#### 3. Response Parser

**Purpose**: LLM response processing and validation

**Key Functions**:

- `extract_reason_and_answer()` - Parse structured responses
- Response validation and error handling

#### 4. Prompt Library

**Purpose**: Battle-tested prompt templates

**Available Prompts**:

- `extract_relation_prompt` - Relation selection guidance
- `answer_prompt` - Final answer generation
- `cot_prompt` - Chain-of-thought reasoning

#### 5. Utils Library

**Purpose**: Common utilities and LLM interface

**Key Functions**:

- `run_llm()` - LLM API calls with error handling
- `break_question()` - Question decomposition
- `save_2_jsonl()` - Result serialization

---

## Evaluation Framework Container Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Evaluation Framework Container                          │
│                                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐       │
│  │                  │     │                  │     │                  │       │
│  │  Dataset Loader  │────▶│  Metric Calculator│────▶│  Report Generator│       │
│  │                  │     │                  │     │                  │       │
│  │ - load_cwq       │     │ - exact_match    │     │ - format_results │       │
│  │ - load_webqsp    │     │ - f1_score       │     │ - compare_systems│       │
│  │ - load_grailqa   │     │ - aggregate_stats│     │ - output_report  │       │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘       │
│                                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐                               │
│  │                  │     │                  │                               │
│  │ Format Converter │     │  Compatibility   │                               │
│  │                  │     │     Layer        │                               │
│  │ - json_to_jsonl  │     │ - pog_format     │                               │
│  │ - jsonl_to_json  │     │ - result_align   │                               │
│  │ - validate_format│     │ - alias_handling │                               │
│  └──────────────────┘     └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. Dataset Loader

**Purpose**: Load and validate benchmark datasets

**Supported Formats**:

- CWQ (ComplexWebQuestions)
- WebQSP (WebQuestionsSP)
- GrailQA

#### 2. Metric Calculator

**Purpose**: Calculate evaluation metrics

**Metrics**:

- Exact Match (EM)
- F1 Score
- Aggregate statistics

#### 3. Report Generator

**Purpose**: Generate evaluation reports and comparisons

#### 4. Format Converter

**Purpose**: Convert between result formats

#### 5. Compatibility Layer

**Purpose**: Ensure compatibility with original PoG evaluation

## Inter-Component Communication

**Data Flow Patterns**:

1. **Command Pattern**: Main Runner → PredictivePlanner → EnhancedExecutor
2. **Observer Pattern**: Memory updates triggered by execution events
3. **Strategy Pattern**: Different planning strategies in PredictivePlanner
4. **Adapter Pattern**: PoG Interface adapts legacy functions for new architecture

**Key Interfaces**:

- Question processing pipeline
- Memory state management
- LLM interaction protocols
- SPARQL query optimization
- Result format compatibility
