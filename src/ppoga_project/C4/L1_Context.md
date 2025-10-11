# C4 Level 1 - System Context

## System Context Diagram

```
                    ┌─────────────────┐
                    │   Researcher    │
                    │   (Person)      │
                    └─────────┬───────┘
                              │
                              │ Asks complex questions
                              │ about knowledge graphs
                              ▼
                    ┌─────────────────┐
                    │     PPoGA       │
                    │    System       │
                    │                 │
                    │ Predictive      │
                    │ Plan-on-Graph   │
                    │ Agent           │
                    └─────────┬───────┘
                              │
                              │ Queries & Retrieves
                              │ semantic information
                              ▼
                    ┌─────────────────┐
                    │   Freebase      │
                    │  Knowledge      │
                    │     Graph       │
                    │                 │
                    │ (External       │
                    │  System)        │
                    └─────────────────┘
                              ▲
                              │
                              │ SPARQL queries
                              │ via HTTP endpoint
                              │
                    ┌─────────────────┐
                    │   Local SPARQL  │
                    │   Endpoint      │
                    │ (localhost:8890)│
                    │                 │
                    │ (External       │
                    │  System)        │
                    └─────────────────┘
```

## System Purpose

**PPoGA (Predictive Plan-on-Graph Agent)** is an advanced question-answering system that combines strategic planning with knowledge graph reasoning to answer complex multi-hop questions.

### Key Responsibilities

1. **Strategic Question Decomposition**: Break down complex questions into actionable sub-objectives
2. **Predictive Planning**: Use LLM-based prediction to guide exploration strategy
3. **Knowledge Graph Reasoning**: Execute SPARQL queries to retrieve relevant information
4. **Adaptive Memory Management**: Maintain 3-layer memory system for efficient reasoning
5. **Answer Generation**: Synthesize discoveries into coherent final answers

## External Dependencies

### 1. Researcher (Primary User)

- **Type**: Person
- **Interaction**: Provides complex multi-hop questions
- **Expectation**: Receives accurate, well-reasoned answers

### 2. Freebase Knowledge Graph

- **Type**: External Data System
- **Technology**: RDF Triple Store
- **Access**: Via SPARQL endpoint (localhost:8890)
- **Content**: Structured factual knowledge about entities and relationships

### 3. LLM Services (OpenAI/Azure)

- **Type**: External AI Service
- **Technology**: REST API
- **Models**: GPT-3.5-turbo, GPT-4, etc.
- **Usage**: Strategic planning, prediction, reasoning, answer generation

## System Boundaries

**In Scope:**

- Question decomposition and strategic planning
- Knowledge graph exploration and reasoning
- Memory management and optimization
- Answer synthesis and evaluation

**Out of Scope:**

- Knowledge graph maintenance or updates
- LLM model training or fine-tuning
- User interface beyond command-line interaction
- Real-time performance requirements

## Success Criteria

1. **Accuracy**: High exact match (EM) scores on benchmark datasets (CWQ, WebQSP, GrailQA)
2. **Efficiency**: Minimal SPARQL queries and LLM calls per question
3. **Scalability**: Handle complex multi-hop questions with deep reasoning chains
4. **Compatibility**: Results comparable to original PoG evaluation framework

## Non-Functional Requirements

- **Performance**: Process questions within reasonable time limits
- **Reliability**: Graceful handling of API failures and missing data
- **Maintainability**: Clear separation of concerns and modular architecture
- **Extensibility**: Easy integration of new planning strategies or optimization techniques
