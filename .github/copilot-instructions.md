# PPoGA Copilot Instructions

## Architecture Overview

**PPoGA** combines Plan-on-Graph (PoG) SPARQL execution with predictive reasoning, using a hybrid architecture:

- `src/ppoga_project/PoG/` - **Core Engine**: Original PoG research code (freebase_func.py for SPARQL, utils.py for LLM calls)
- `src/ppoga_project/ppoga/` - **PPoGA Extensions**: predictive_planner.py, enhanced_executor.py, enhanced_memory.py
- `src/ppoga_project/main_ppoga.py` - **Main orchestrator** (529 lines) bridges PoG and PPoGA components

### Critical Import Pattern

PoG module uses **relative imports** (`.utils`, `.freebase_func`), while PPoGA components use **sys.path manipulation**:

```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PoG.utils import run_llm
```

### Three-Layer Memory Architecture

1. **Strategic Layer**: Plans and high-level decisions (`PPoGAMemory.strategic_plan`)
2. **Execution Layer**: Prediction-Action-Observation-Thought cycles (`ExecutionCycle` dataclass)
3. **Knowledge Layer**: Entities, relations, reasoning paths discovered during execution

## Development Workflow

### Environment Setup

Use **standard .venv** (not Poetry). Key setup command:

```bash
./dev.sh test  # Auto-creates .venv, installs deps, runs tests
```

### Testing Framework (TDD-focused)

- **32 test cases** across unit/integration/benchmark
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.benchmark`
- Mock fixtures in `tests/fixtures/mock_responses.py` for LLM/SPARQL responses
- Configuration: `pytest.ini` with coverage reporting, strict markers

### Key Commands

```bash
./dev.sh test-unit        # Unit tests only
./dev.sh test-integration # Integration tests
./dev.sh lint            # flake8, black, isort, mypy
./dev.sh format          # Auto-format with black/isort
./dev.sh coverage        # HTML coverage report
```

## Integration Points

### PoG-PPoGA Bridge

**Critical**: PoG's `run_llm()` has typo in parameter name: `opeani_api_keys` (not `openai_api_keys`)

### SPARQL Execution

- Requires **local Freebase** at `http://localhost:8890/sparql`
- PoG functions: `entity_search()`, `relation_search_prune()`, `provide_triple()`
- Results format: List of [entity, relation, entity] triples

### Configuration System

`config.py` uses dataclass pattern with environment variable fallbacks:

```python
config = PPoGAConfig.from_env()  # Loads from .env file
```

## Project-Specific Patterns

### Error Handling Strategy

- **Two-level correction**: Tactical (step-level) and Strategic (plan-level)
- Use `PPoGAMemory.add_execution_cycle()` to track all attempts
- PreAct prediction vs actual observation comparison drives correction logic

### Dataset Processing

- Supports CWQ, WebQSP, GrailQA with entity aliases in `cope_alias/`
- Results saved as timestamped JSON in `results/` directory
- Use `prepare_dataset()` from PoG utils for consistent data loading

### File Organization Convention

- Active code: `src/ppoga_project/` only
- All v2 references removed (systematic renaming completed)
- Test structure mirrors src structure: `tests/unit/test_planner.py` ↔ `src/ppoga_project/ppoga/predictive_planner.py`

### Code Modification Guidelines

- **Minimize PoG changes**: The `src/ppoga_project/PoG/` directory contains original research code - modify only when absolutely necessary
- **C4 Documentation**: ALL code changes must be reflected in `src/ppoga_project/C4/` architecture documentation
  - Update relevant C4 level files (L1_Context.md, L2_Container.md, L3_Component.md, L4_Code.md)
  - Maintain consistency between implementation and architectural documentation

### LLM Integration Pattern

Always use PoG's `run_llm()` wrapper for consistency:

```python
response, tokens = run_llm(prompt, temperature, max_tokens, api_key, engine)
```

## Debugging & Troubleshooting

### Common Issues

1. **Import errors**: Check sys.path manipulation and relative import syntax
2. **API key errors**: Parameter name is `opeani_api_keys` (typo in PoG code)
3. **SPARQL failures**: Verify Freebase endpoint accessibility
4. **Memory overflow**: Use `max_iterations` and `max_depth` limits in config
5. **PoG modifications**: Avoid changing original PoG code unless critical - prefer extending in ppoga/ directory

### Code Change Protocol

1. **Before modifying**: Check if change can be implemented in `ppoga/` extension layer instead of PoG
2. **After any change**: Update corresponding C4 documentation in `src/ppoga_project/C4/`
3. **Documentation sync**: Ensure architectural docs reflect current implementation state

### Development State

- **Test infrastructure**: Complete TDD framework with 32 test skeletons
- **CI/CD**: GitHub Actions with multi-Python version testing
- **Code quality**: Configured black, isort, flake8, mypy in pyproject.toml
