# Original PoG - C4 Model Documentation

C4 Model documentation for the original PoG (Plan-on-Graph) system.

## C4 Model Levels

1. **Level 1 - Context**: System context and external dependencies
2. **Level 2 - Container**: High-level technology choices and system boundaries
3. **Level 3 - Component**: Internal component relationships within containers
4. **Level 4 - Code**: Implementation details and function structures

## Documentation Files

- `L1_Context.md` - System Context diagram and description
- `L2_Container.md` - Container diagram and technology stack
- `L3_Component.md` - Component diagrams for each container
- `L4_Code.md` - Code-level architecture and function details

## Overview

Original PoG is a proven knowledge graph reasoning system that features:

- **Efficient SPARQL Operations**: Optimized knowledge graph querying
- **Intelligent Pruning**: LLM-guided relation and entity selection
- **Memory Management**: File-based memory system for multi-hop reasoning
- **Proven Performance**: Battle-tested on CWQ, WebQSP, and GrailQA datasets

## System Files

- `main_freebase.py` - Main execution script
- `freebase_func.py` - SPARQL operations and KG reasoning (19 functions)
- `utils.py` - LLM utilities and response processing (14 functions)
- `prompt_list.py` - Comprehensive prompt library (7 specialized prompts)

Created: October 11, 2025
