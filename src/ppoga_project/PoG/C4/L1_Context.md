# C4 Level 1 - System Context (Original PoG)

## System Context Diagram

```
                    ┌─────────────────┐
                    │   Researcher    │
                    │   (Person)      │
                    └─────────┬───────┘
                              │
                              │ Asks complex multi-hop
                              │ questions about KG
                              ▼
                    ┌─────────────────┐
                    │   Original PoG  │
                    │     System      │
                    │                 │
                    │ Plan-on-Graph   │
                    │ Reasoning Agent │
                    └─────────┬───────┘
                              │
                              │ Executes optimized
                              │ SPARQL queries
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
                              │ HTTP/SPARQL Protocol
                              │ (localhost:8890)
                              │
                    ┌─────────────────┐
                    │   Local SPARQL  │
                    │   Endpoint      │
                    │   (Virtuoso)    │
                    │                 │
                    │ (External       │
                    │  System)        │
                    └─────────────────┘

        ┌─────────────────┐                    ┌─────────────────┐
        │   OpenAI API    │                    │   Sentence      │
        │    Service      │◄───────────────────┤  Transformers   │
        │                 │                    │    Library      │
        │ GPT-3.5/GPT-4   │                    │                 │
        │ (External)      │                    │ (External)      │
        └─────────────────┘                    └─────────────────┘
                ▲                                        ▲
                │                                        │
                │ LLM API calls                         │ Similarity search
                │ for reasoning                         │ for entity ranking
                │                                        │
                └────────────────────┬───────────────────┘
                                     │
                                     ▼
                           ┌─────────────────┐
                           │   Original PoG  │
                           │     System      │
                           └─────────────────┘
```

## System Purpose

**Original PoG (Plan-on-Graph)** is a knowledge graph reasoning system that efficiently answers complex multi-hop questions through strategic planning and optimized exploration.

### Key Responsibilities

1. **Question Decomposition**: Break complex questions into sub-objectives
2. **Strategic Planning**: Plan exploration steps to minimize unnecessary queries
3. **Optimized SPARQL Execution**: Execute efficient knowledge graph queries
4. **Intelligent Pruning**: Use LLM guidance to filter irrelevant relations/entities
5. **Memory Management**: Maintain exploration state across reasoning steps
6. **Early Stopping**: Detect when sufficient information is gathered
7. **Answer Generation**: Synthesize final answers from discovered knowledge

## External Dependencies

### 1. Researcher (Primary User)

- **Type**: Person
- **Interaction**: Provides complex multi-hop questions via command line
- **Input Format**: Natural language questions
- **Expected Output**: Accurate factual answers with reasoning traces

### 2. Freebase Knowledge Graph

- **Type**: External RDF Triple Store
- **Technology**: SPARQL endpoint (Virtuoso)
- **Access Method**: HTTP queries to localhost:8890/sparql
- **Content**: 3+ billion facts about entities, relations, and properties
- **Format**: RDF triples with Freebase schema

### 3. OpenAI API Service

- **Type**: External Language Model Service
- **Technology**: REST API
- **Models Used**: GPT-3.5-turbo, GPT-4
- **Usage Patterns**:
  - Question decomposition into sub-objectives
  - Relation selection and pruning
  - Entity relevance assessment
  - Answer generation and validation
  - Memory summarization

### 4. Sentence Transformers Library

- **Type**: External ML Library
- **Technology**: Python package
- **Purpose**: Semantic similarity computation
- **Usage**: Entity ranking and relevance scoring

## System Boundaries

**In Scope:**

- Multi-hop question answering on knowledge graphs
- SPARQL query optimization and execution
- LLM-guided exploration strategies
- Memory management for complex reasoning
- Performance optimization (early stopping, pruning)
- Batch processing of question datasets

**Out of Scope:**

- Knowledge graph construction or updates
- LLM model training or fine-tuning
- Real-time interactive question-answering interfaces
- Multi-modal reasoning (text + images)
- Temporal reasoning or knowledge graph versioning

## Success Criteria

1. **Accuracy Metrics**:

   - **Exact Match (EM)**: Target >65% on ComplexWebQuestions
   - **F1 Score**: High token-level overlap with ground truth
   - **Precision**: Minimize false positive answers

2. **Efficiency Metrics**:

   - **SPARQL Queries**: Minimize unnecessary graph exploration
   - **LLM Calls**: Optimize API usage through intelligent caching
   - **Execution Time**: Complete questions within reasonable time limits

3. **Robustness**:
   - Handle edge cases and missing information gracefully
   - Robust error handling for API failures
   - Consistent performance across different question types

## Non-Functional Requirements

### Performance

- **Response Time**: Complete most questions within 2-5 minutes
- **Scalability**: Process datasets with thousands of questions
- **Resource Usage**: Efficient memory and API usage

### Reliability

- **Error Handling**: Graceful degradation when external services fail
- **Data Consistency**: Maintain exploration state integrity
- **Reproducibility**: Deterministic results for same inputs

### Maintainability

- **Code Structure**: Clear functional decomposition
- **Documentation**: Comprehensive inline documentation
- **Testing**: Evaluation framework for performance validation

### Security

- **API Keys**: Secure handling of external service credentials
- **Data Privacy**: No persistent storage of sensitive information
- **Network Security**: Secure communication with external services

## Quality Attributes

### Efficiency

- Proven optimization techniques for knowledge graph exploration
- Intelligent early stopping to prevent unnecessary computation
- LLM-guided pruning to focus on relevant information

### Accuracy

- Battle-tested prompts for various reasoning tasks
- Memory system to maintain context across exploration steps
- Multi-layered validation of answers before finalization

### Extensibility

- Modular design allowing easy addition of new optimization techniques
- Configurable parameters for different datasets and domains
- Clear interfaces for integrating new reasoning strategies
