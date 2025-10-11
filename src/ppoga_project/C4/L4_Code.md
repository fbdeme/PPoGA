# C4 Level 4 - Code Architecture

## Class Structure and Implementation Details

### PPoGA Core - Class Diagrams

#### 1. PPoGAMemory Class

```python
@dataclass
class ExecutionCycle:
    step_id: int
    prediction: Dict[str, Any]
    action: Dict[str, Any]
    observation: str
    thought: Dict[str, Any]
    timestamp: float

@dataclass
class PlanStep:
    step_id: int
    description: str
    objective: str
    expected_outcome: str
    status: str = "not_started"

class PPoGAMemory:
    """3-layer memory architecture"""

    # Strategic Layer
    strategy: Dict = {
        "initial_question": str,
        "overall_plan": List[PlanStep],
        "plan_rationale": str,
        "status": str,
        "alternative_plans": List[Dict],
        "replan_count": int
    }

    # Execution Layer
    execution: Dict = {
        "current_step_id": int,
        "cycles": Dict[int, List[ExecutionCycle]],
        "total_cycles": int,
        "current_prediction": Dict,
        "current_observation": str,
        "current_thought": Dict
    }

    # Knowledge Layer
    knowledge: Dict = {
        "discovered_entities": Dict[str, str],
        "explored_relations": Dict[str, List[str]],
        "exploration_graph": Dict[str, Dict[str, List[str]]],
        "reasoning_chains": List[List[Tuple[str, str, str]]],
        "candidates": List[str]
    }

    # Methods
    def update_plan(plan_steps: List[Dict], rationale: str)
    def get_current_step() -> Optional[PlanStep]
    def advance_step() -> bool
    def add_execution_cycle(step_id, prediction, action, observation, thought)
    def add_discovered_entities(entities: Dict[str, str])
    def log_llm_interaction(call_type, prompt, response, success, metadata)
    def add_exploration_result(entity_id, relation, target_entities)
    def get_memory_summary() -> Dict[str, Any]
    def get_context_for_llm() -> str
```

#### 2. PredictivePlanner Class

```python
class PredictivePlanner:
    """Strategic planning with predictive reasoning"""

    # Configuration
    llm_config: Dict[str, Any]

    # Core Planning Methods
    def decompose_plan(question: str, context: Dict) -> Dict[str, Any]:
        """Break question into strategic steps"""

    def predict_step_outcome(
        step: Dict,
        current_context: Dict,
        memory_context: str
    ) -> Dict[str, Any]:
        """Predict execution results before action"""

    def observe(
        action_result: Dict,
        current_context: Dict,
        memory_context: str
    ) -> Dict[str, Any]:
        """Process and analyze execution results"""

    def think_and_evaluate(
        prediction: Dict,
        observation: Dict,
        memory_context: str
    ) -> Dict[str, Any]:
        """Evaluate prediction vs reality and learn"""

    # Advanced Planning
    def strategic_replan(
        current_progress: Dict,
        obstacles: List[str],
        memory_context: str
    ) -> Dict[str, Any]:
        """Dynamic replanning when obstacles encountered"""

    def final_answer_synthesis(
        all_discoveries: List[Dict],
        memory_context: str
    ) -> Dict[str, Any]:
        """Synthesize final answer from discoveries"""

    # Internal LLM Interaction
    def _call_llm(prompt: str, temperature: float = 0.3) -> Tuple[str, Dict]
    def _parse_structured_response(response: str) -> Dict[str, Any]
```

#### 3. EnhancedExecutor Class

```python
class EnhancedExecutor:
    """Bridge between planning and knowledge graph operations"""

    # Configuration
    kg_config: Dict[str, Any]
    stats: Dict[str, int]

    # Core Execution
    def execute_step(
        entity_id: str,
        entity_name: str,
        plan_step: Dict[str, Any],
        args: Any
    ) -> Dict[str, Any]:
        """Execute individual plan step"""

    # Knowledge Graph Operations (via PoG Legacy)
    def _perform_relation_search(
        entity_id: str,
        sub_questions: List[str],
        entity_name: str,
        args: Any
    ) -> Tuple[List[Dict], Dict]:
        """Use PoG's relation_search_prune"""

    def _perform_entity_search(
        entity_id: str,
        relation: str,
        head: bool
    ) -> List[str]:
        """Use PoG's entity_search"""

    def _format_triples(
        entity_candidates_id: List[str],
        relation: str
    ) -> Tuple[List[str], List[str]]:
        """Use PoG's provide_triple"""

    # Result Processing
    def _process_exploration_results(
        retrieved_relations: List[Dict],
        entity_id: str,
        entity_name: str
    ) -> Dict[str, Any]:
        """Process and structure KG exploration results"""

    def _update_statistics(success: bool, entities_count: int):
        """Update execution statistics"""
```

### PoG Legacy - Function Architecture

#### 1. SPARQL Engine (freebase_func.py)

```python
# Core SPARQL Operations
def execute_sparql(sparql_query: str) -> List[Dict]:
    """Execute SPARQL query against Freebase endpoint"""

def abandon_rels(relation: str) -> bool:
    """Filter unnecessary relations"""

def replace_relation_prefix(relations: List[Dict]) -> List[str]:
    """Clean relation URIs"""

def replace_entities_prefix(entities: List[Dict]) -> List[str]:
    """Clean entity URIs"""

# Entity and Relation Operations
def id2entity_name_or_type(entity_id: str) -> str:
    """Convert entity ID to human-readable name"""

def entity_search(entity: str, relation: str, head: bool = True) -> List[str]:
    """Search for entities via specific relation"""

def relation_search_prune(
    entity_id: str,
    sub_questions: List[str],
    entity_name: str,
    pre_relations: List[str],
    pre_head: bool,
    question: str,
    args: Any
) -> Tuple[List[Dict], Dict]:
    """LLM-guided relation search with pruning"""

def provide_triple(
    entity_candidates_id: List[str],
    relation: str
) -> Tuple[List[str], List[str]]:
    """Process and format triples"""

# SPARQL Query Templates
sparql_head_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation WHERE { ns:%s ?relation ?x . }"""

sparql_tail_relations = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation WHERE { ?x ?relation ns:%s . }"""

sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity WHERE { ns:%s ns:%s ?tailEntity . }"""

sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity WHERE { ?tailEntity ns:%s ns:%s . }"""
```

#### 2. Utils Library (utils.py)

```python
# LLM Interface
def run_llm(
    prompt: str,
    temperature: float,
    max_tokens: int,
    api_keys: str,
    engine: str,
    print_in: bool = True,
    print_out: bool = True
) -> Tuple[str, Dict]:
    """Core LLM API interface with error handling"""

# Response Processing
def extract_reason_and_answer(string: str) -> Tuple[str, str, str]:
    """Parse structured LLM responses"""

def break_question(question: str, args: Any) -> Tuple[str, Dict]:
    """Question decomposition using LLM"""

# File Operations
def save_2_jsonl(
    question: str,
    question_string: str,
    answer: str,
    cluster_chain_of_entities: List,
    call_num: int,
    all_t: Dict,
    start_time: float,
    file_name: str
):
    """Save results in JSONL format"""

def convert_dict_name(
    ent_rel_ent_dict: Dict,
    entid_name: Dict[str, str]
) -> Dict:
    """Convert entity IDs to names in result dictionary"""
```

#### 3. Prompt Library (prompt_list.py)

```python
# Core Prompts
subobjective_prompt = """Please break down the process of answering
the question into as few subobjectives as possible..."""

extract_relation_prompt = """Please provide as few highly relevant
relations as possible to the question..."""

answer_prompt = """Given a question and the associated retrieved
knowledge graph triplets, you are asked to answer..."""

cot_prompt = """Please answer the question according to your
knowledge step by step..."""

# Prompt Categories:
# 1. Planning Prompts - Question decomposition
# 2. Exploration Prompts - Relation/entity selection
# 3. Reasoning Prompts - Answer generation
# 4. Evaluation Prompts - Sufficiency assessment
```

### Evaluation Framework - Class Structure

#### 1. Evaluation Engine (eval.py)

```python
class EvaluationEngine:
    """Main evaluation orchestrator"""

    def __init__(self, dataset_name: str, output_file: str):
        self.dataset_name = dataset_name
        self.output_file = output_file

    def run_evaluation(self) -> Dict[str, float]:
        """Execute full evaluation pipeline"""

    def calculate_metrics(
        ground_truth: List[Dict],
        predictions: List[Dict]
    ) -> Dict[str, float]:
        """Calculate EM, F1, and other metrics"""

    def generate_report(self, metrics: Dict) -> str:
        """Generate detailed evaluation report"""

# Core Evaluation Functions
def prepare_dataset_for_eval(
    dataset_name: str,
    output_file: str
) -> Tuple[List[Dict], str, List[Dict]]:
    """Load datasets and results for evaluation"""

def align(
    dataset_name: str,
    question_string: str,
    data: Dict,
    ground_truth_datas: List[Dict],
    aname_dict: Dict,
    alias_dict: Dict,
    add_ans_alias_dict: Dict
) -> Tuple[List[str], Dict]:
    """Align predicted answers with ground truth"""
```

#### 2. Format Converter (convert_results.py)

```python
class ResultConverter:
    """Convert between result formats"""

    def json_to_jsonl(
        input_file: str,
        output_file: str,
        question_field: str = "question"
    ) -> None:
        """Convert JSON results to JSONL format"""

    def jsonl_to_json(
        input_file: str,
        output_file: str
    ) -> None:
        """Convert JSONL results to JSON format"""

    def validate_format(file_path: str) -> bool:
        """Validate result file format"""

    def extract_pog_format(result_dict: Dict) -> Dict:
        """Extract PoG-compatible format from PPoGA results"""
```

## Key Implementation Patterns

### 1. Strategy Pattern

**Used in**: PredictivePlanner prompt selection

```python
def _select_prompt_strategy(self, task_type: str) -> str:
    strategies = {
        "decompose": self.decompose_plan_prompt,
        "predict": self.predict_outcome_prompt,
        "observe": self.observe_prompt,
        "evaluate": self.think_evaluate_prompt
    }
    return strategies.get(task_type, self.default_prompt)
```

### 2. Observer Pattern

**Used in**: Memory updates triggered by execution events

```python
def add_execution_cycle(self, step_id, prediction, action, observation, thought):
    # Update execution layer
    cycle = ExecutionCycle(step_id, prediction, action, observation, thought)
    self.execution["cycles"][step_id].append(cycle)

    # Notify observers (statistics, logging, etc.)
    self._notify_observers("execution_cycle_added", cycle)
```

### 3. Adapter Pattern

**Used in**: PoG Legacy integration

```python
class PoGAdapter:
    """Adapt PoG functions for PPoGA architecture"""

    def adapted_relation_search(self, entity_id, context, args):
        # Convert PPoGA context to PoG parameters
        sub_questions = self._extract_subobjectives(context)
        entity_name = self._get_entity_name(entity_id)

        # Call original PoG function
        return relation_search_prune(
            entity_id, sub_questions, entity_name,
            [], False, context["question"], args
        )
```

### 4. Template Method Pattern

**Used in**: Execution pipeline

```python
def execute_step(self, entity_id, entity_name, plan_step, args):
    # Template method with fixed algorithm
    self._pre_execution_setup()
    result = self._perform_kg_operations(entity_id, plan_step, args)
    self._post_execution_processing(result)
    return self._format_execution_result(result)
```

## Error Handling and Resilience

### 1. LLM API Failures

```python
def _call_llm_with_retry(self, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return self._call_llm(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                return self._fallback_response(prompt)
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 2. SPARQL Endpoint Failures

```python
def execute_sparql(sparql_query):
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        return sparql.query().convert()
    except Exception as e:
        print(f"SPARQL error: {e}")
        return {"results": {"bindings": []}}  # Empty result
```

### 3. Memory State Corruption

```python
def add_execution_cycle(self, *args):
    try:
        # Validate input parameters
        self._validate_cycle_data(*args)
        # Perform update
        self._update_execution_state(*args)
    except Exception as e:
        # Log error and maintain consistent state
        self._log_error(f"Memory update failed: {e}")
        self._restore_last_known_good_state()
```

## Performance Optimizations

### 1. Lazy Loading

```python
@property
def exploration_graph(self):
    if not hasattr(self, '_exploration_graph'):
        self._exploration_graph = self._build_exploration_graph()
    return self._exploration_graph
```

### 2. Caching

```python
@lru_cache(maxsize=1000)
def id2entity_name_or_type(entity_id: str) -> str:
    # Cache entity name lookups
    return self._fetch_entity_name(entity_id)
```

### 3. Batch Processing

```python
def batch_entity_search(self, entity_relation_pairs):
    # Group by relation for efficient batch queries
    batched_queries = self._group_by_relation(entity_relation_pairs)
    results = {}
    for relation, entities in batched_queries.items():
        batch_result = self._execute_batch_query(relation, entities)
        results.update(batch_result)
    return results
```

## Recent Code Architecture Changes

### Directory Structure Update (v2 → Official)

**Completed Systematic Renaming**:

- `ppoga_v2/` → `ppoga/` (main component directory)
- `main_ppoga_v2.py` → `main_ppoga.py` (primary execution script)
- All function names: `run_ppoga_v2` → `run_ppoga`
- All documentation references: "PPoGA v2" → "PPoGA"

**Import Path Corrections**:

```python
# PoG module - fixed relative imports
from .utils import *
from .freebase_func import *
from .prompt_list import *

# PPoGA components - sys.path manipulation pattern
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PoG.utils import run_llm
```

### Test-Driven Development Infrastructure

**Complete TDD Framework** (32 test cases):

```
tests/
├── conftest.py                 # Global fixtures and configuration
├── fixtures/
│   ├── mock_responses.py       # LLM/SPARQL response mocks
│   └── sample_questions.py     # Test data sets
├── unit/
│   ├── test_planner.py         # PredictivePlanner tests
│   ├── test_executor.py        # EnhancedExecutor tests
│   └── test_memory.py          # PPoGAMemory tests
├── integration/
│   └── test_ppoga_flow.py      # End-to-end workflow tests
└── benchmark/
    └── test_performance.py     # Performance benchmarks
```

**Development Tooling**:

- `./dev.sh` - Primary development helper script
- `pytest.ini` - Comprehensive test configuration with markers
- `pyproject.toml` - Code quality tools (black, isort, flake8, mypy)
- GitHub Actions CI/CD with multi-Python version testing

### Environment Management

**Standardized on .venv** (Poetry removed):

```bash
# Setup pattern
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Configuration Management**:

```python
# Dataclass pattern with environment fallbacks
@dataclass
class PPoGAConfig:
    @classmethod
    def from_env(cls) -> "PPoGAConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            sparql_endpoint=os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql"),
            # ... other config
        )
```

### Critical Integration Fixes

**PoG LLM Interface** (Preserved original typo):

```python
# Parameter name maintained for compatibility
def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo"):
    # Original PoG implementation preserved
```

**Memory Architecture Optimization**:

- Strategic Layer: Plan persistence and replanning logic
- Execution Layer: Prediction-Action-Observation-Thought tracking
- Knowledge Layer: Entity/relation discovery with graph building
