"""
Mock responses for testing PPoGA system components
"""

import json
from typing import Dict, Any, List

# Mock LLM responses for different scenarios
MOCK_LLM_RESPONSES = {
    "planning": {
        "decompose_simple": {
            "content": json.dumps(
                {
                    "plan": [
                        {
                            "step_id": 1,
                            "description": "Find the entity 'The Godfather' in the knowledge graph",
                            "action": "entity_search",
                            "target": "The Godfather",
                        },
                        {
                            "step_id": 2,
                            "description": "Find the director relation for 'The Godfather'",
                            "action": "relation_search",
                            "target": "film.film.directed_by",
                        },
                        {
                            "step_id": 3,
                            "description": "Extract the director name",
                            "action": "extract_answer",
                            "target": "director",
                        },
                    ],
                    "strategy": "direct_lookup",
                    "estimated_hops": 1,
                }
            ),
            "role": "assistant",
        },
        "decompose_complex": {
            "content": json.dumps(
                {
                    "plan": [
                        {
                            "step_id": 1,
                            "description": "Find 'The Godfather' movie entity",
                            "action": "entity_search",
                            "target": "The Godfather",
                        },
                        {
                            "step_id": 2,
                            "description": "Find the director of 'The Godfather'",
                            "action": "relation_search",
                            "target": "film.film.directed_by",
                        },
                        {
                            "step_id": 3,
                            "description": "Find spouse relationship for the director",
                            "action": "relation_search",
                            "target": "people.person.spouse_s",
                        },
                        {
                            "step_id": 4,
                            "description": "Extract spouse name",
                            "action": "extract_answer",
                            "target": "spouse",
                        },
                    ],
                    "strategy": "multi_hop_traversal",
                    "estimated_hops": 3,
                }
            ),
            "role": "assistant",
        },
    },
    "prediction": {
        "step_outcome": {
            "content": json.dumps(
                {
                    "prediction": {
                        "expected_entities": ["The Godfather", "m.047z4b5"],
                        "expected_relations": ["film.film.directed_by"],
                        "confidence": 0.9,
                        "reasoning": "This should find The Godfather movie entity with high confidence",
                    }
                }
            ),
            "role": "assistant",
        }
    },
    "evaluation": {
        "prediction_vs_reality": {
            "content": json.dumps(
                {
                    "evaluation": {
                        "prediction_accuracy": 0.85,
                        "unexpected_findings": [
                            "Found additional relation: film.film.starring"
                        ],
                        "learning": "Movie entities often have multiple relation types",
                        "adjustment_needed": False,
                    }
                }
            ),
            "role": "assistant",
        }
    },
    "replanning": {
        "strategic_replan": {
            "content": json.dumps(
                {
                    "replan": {
                        "reason": "Previous path blocked - entity not found",
                        "new_strategy": "try_alternative_entity_name",
                        "modified_steps": [
                            {
                                "step_id": 1,
                                "description": "Search for alternative names of 'The Godfather'",
                                "action": "entity_search_alternative",
                                "target": "The Godfather OR Godfather OR The Godfather (1972)",
                            }
                        ],
                        "confidence": 0.7,
                    }
                }
            ),
            "role": "assistant",
        }
    },
}

# Mock SPARQL query responses
MOCK_SPARQL_RESPONSES = {
    "entity_search": {
        "the_godfather": {
            "results": {
                "bindings": [
                    {
                        "entity": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.047z4b5",
                        },
                        "name": {"type": "literal", "value": "The Godfather"},
                    }
                ]
            }
        },
        "francis_coppola": {
            "results": {
                "bindings": [
                    {
                        "entity": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.0c4dn",
                        },
                        "name": {"type": "literal", "value": "Francis Ford Coppola"},
                    }
                ]
            }
        },
    },
    "relation_search": {
        "director_relation": {
            "results": {
                "bindings": [
                    {
                        "subject": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.047z4b5",
                        },
                        "relation": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/film.film.directed_by",
                        },
                        "object": {
                            "type": "uri",
                            "value": "http://rdf.freebase.com/ns/m.0c4dn",
                        },
                    }
                ]
            }
        }
    },
}

# Mock knowledge graph triplets
MOCK_TRIPLETS = {
    "the_godfather": [
        ["The Godfather", "film.film.directed_by", "Francis Ford Coppola"],
        ["The Godfather", "film.film.starring", "Marlon Brando"],
        ["The Godfather", "film.film.starring", "Al Pacino"],
        ["The Godfather", "film.film.release_date", "1972-03-24"],
    ],
    "francis_coppola": [
        ["Francis Ford Coppola", "people.person.spouse_s", "m.marriage1"],
        ["m.marriage1", "people.marriage.spouse", "Eleanor Coppola"],
        ["Francis Ford Coppola", "film.director.film", "The Godfather"],
        ["Francis Ford Coppola", "film.director.film", "Apocalypse Now"],
    ],
}

# Mock memory states for testing
MOCK_MEMORY_STATES = {
    "initial": {
        "strategy": {
            "initial_question": "Who directed The Godfather?",
            "overall_plan": [],
            "plan_rationale": "",
            "status": "planning",
            "alternative_plans": [],
            "replan_count": 0,
        },
        "execution": {
            "current_step_id": 0,
            "cycles": {},
            "total_cycles": 0,
            "current_prediction": {},
            "current_observation": "",
            "current_thought": {},
        },
        "knowledge": {
            "discovered_entities": {},
            "explored_relations": {},
            "exploration_graph": {},
            "reasoning_chains": [],
            "candidates": [],
        },
    },
    "mid_execution": {
        "strategy": {
            "initial_question": "Who directed The Godfather?",
            "overall_plan": [
                {
                    "step_id": 1,
                    "description": "Find The Godfather entity",
                    "status": "completed",
                },
                {
                    "step_id": 2,
                    "description": "Find director relation",
                    "status": "in_progress",
                },
            ],
            "plan_rationale": "Direct entity-relation lookup strategy",
            "status": "executing",
            "alternative_plans": [],
            "replan_count": 0,
        },
        "execution": {
            "current_step_id": 2,
            "cycles": {1: [{"prediction": {}, "action": {}, "observation": {}}]},
            "total_cycles": 1,
            "current_prediction": {"expected_entities": ["The Godfather"]},
            "current_observation": "Found entity: The Godfather (m.047z4b5)",
            "current_thought": {"status": "proceeding", "confidence": 0.9},
        },
        "knowledge": {
            "discovered_entities": {"The Godfather": "m.047z4b5"},
            "explored_relations": {},
            "exploration_graph": {"m.047z4b5": {}},
            "reasoning_chains": [],
            "candidates": [],
        },
    },
}


def get_mock_llm_response(response_type: str, scenario: str) -> Dict[str, Any]:
    """Get a mock LLM response for testing"""
    return MOCK_LLM_RESPONSES.get(response_type, {}).get(scenario, {})


def get_mock_sparql_response(query_type: str, entity: str) -> Dict[str, Any]:
    """Get a mock SPARQL response for testing"""
    return MOCK_SPARQL_RESPONSES.get(query_type, {}).get(entity, {})


def get_mock_triplets(entity: str) -> List[List[str]]:
    """Get mock triplets for an entity"""
    return MOCK_TRIPLETS.get(entity, [])


def get_mock_memory_state(state_name: str) -> Dict[str, Any]:
    """Get a mock memory state for testing"""
    return MOCK_MEMORY_STATES.get(state_name, {})
