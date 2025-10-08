"""
PoG Freebase Functions - Adapted for PPoGA
Enhanced with Azure OpenAI integration and PPoGA compatibility

Original from: https://github.com/liyichen-cly/PoG
Enhanced for PPoGA with Azure OpenAI support
"""

from SPARQLWrapper import SPARQLWrapper, JSON
import random
import json
import time
import re
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# SPARQL configuration - 로컬 Freebase 설정 (추후 실제 Freebase 배포 시 변경)
SPARQLPATH = "http://localhost:8890/sparql"  # Freebase 로컬 배포 주소

# Pre-defined SPARQL queries
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}"""
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


def abandon_rels(relation: str) -> bool:
    """Check if a relation should be abandoned (filtered out)"""
    if (
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameAs" in relation
    ):
        return True
    return False


def execute_sparql(sparql_query: str) -> List[Dict]:
    """Execute SPARQL query against Freebase"""
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"SPARQL Error: {e}")
        return []


def replace_relation_prefix(relations: List[Dict]) -> List[str]:
    """Remove Freebase namespace prefix from relations"""
    return [
        relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for relation in relations
    ]


def replace_entities_prefix(entities: List[Dict]) -> List[str]:
    """Remove Freebase namespace prefix from entities"""
    return [
        entity["tailEntity"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for entity in entities
    ]


def id2entity_name_or_type(entity_id: str) -> str:
    """Convert entity ID to human-readable name"""
    try:
        sparql_query = sparql_id % (entity_id, entity_id)
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if len(results["results"]["bindings"]) == 0:
            return entity_id
        else:
            return results["results"]["bindings"][0]["tailEntity"]["value"]
    except:
        return entity_id


def select_relations(
    string: str, entity_id: str, head_relations: List[str], tail_relations: List[str]
) -> Tuple[bool, List[Dict]]:
    """Select relevant relations from LLM output"""
    try:
        last_brace_l = string.rfind("[")
        last_brace_r = string.rfind("]")

        if last_brace_l < last_brace_r:
            string = string[last_brace_l : last_brace_r + 1]

        relations = []
        rel_list = eval(string.strip())
        for relation in rel_list:
            if relation in head_relations:
                relations.append(
                    {"entity": entity_id, "relation": relation, "head": True}
                )
            elif relation in tail_relations:
                relations.append(
                    {"entity": entity_id, "relation": relation, "head": False}
                )

        if not relations:
            return False, []
        return True, relations
    except:
        return False, []


def construct_relation_prune_prompt(
    question: str,
    sub_questions: List[str],
    entity_name: str,
    total_relations: List[str],
) -> str:
    """Construct prompt for relation pruning"""
    from .pog_prompts import extract_relation_prompt

    return (
        extract_relation_prompt
        + question
        + "\nSubobjectives: "
        + str(sub_questions)
        + "\nTopic Entity: "
        + entity_name
        + "\nRelations: "
        + "; ".join(total_relations)
    )


def relation_search_prune(
    entity_id: str,
    sub_questions: List[str],
    entity_name: str,
    pre_relations: List[str],
    pre_head: bool,
    question: str,
    azure_config: Dict[str, Any],
) -> Tuple[List[Dict], Dict[str, int]]:
    """Search and prune relations for an entity using Azure OpenAI"""
    # Get head and tail relations from Freebase
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execute_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)

    sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
    tail_relations = execute_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    # Filter unnecessary relations
    head_relations = [
        relation for relation in head_relations if not abandon_rels(relation)
    ]
    tail_relations = [
        relation for relation in tail_relations if not abandon_rels(relation)
    ]

    # Remove previously used relations
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()

    # Create prompt and call Azure OpenAI
    prompt = construct_relation_prune_prompt(
        question, sub_questions, entity_name, total_relations
    )

    try:
        from ..ppoga_core.azure_llm import call_azure_openai

        result, token_num = call_azure_openai(prompt, azure_config, temperature=0.3)

        flag, retrieve_relations = select_relations(
            result, entity_id, head_relations, tail_relations
        )

        if flag:
            return retrieve_relations, token_num
        else:
            return [], token_num
    except Exception as e:
        print(f"Error in relation_search_prune: {e}")
        return [], {"total": 0, "input": 0, "output": 0}


def entity_search(entity: str, relation: str, head: bool = True) -> List[str]:
    """Search for entities connected by a relation"""
    try:
        if head:
            tail_entities_extract = sparql_tail_entities_extract % (entity, relation)
            entities = execute_sparql(tail_entities_extract)
        else:
            head_entities_extract = sparql_head_entities_extract % (relation, entity)
            entities = execute_sparql(head_entities_extract)

        entity_ids = replace_entities_prefix(entities)
        return entity_ids
    except Exception as e:
        print(f"Error in entity_search: {e}")
        return []


def provide_triple(
    entity_candidates_id: List[str], relation: str
) -> Tuple[List[str], List[str]]:
    """Convert entity IDs to names and organize triples"""
    entity_candidates = []
    for entity_id in entity_candidates_id:
        if entity_id.startswith("m."):
            entity_candidates.append(id2entity_name_or_type(entity_id))
        else:
            entity_candidates.append(entity_id)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id

    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
    entity_candidates, entity_candidates_id = list(ent_id_dict.keys()), list(
        ent_id_dict.values()
    )
    return entity_candidates, entity_candidates_id


def update_history(
    entity_candidates: List[str],
    ent_rel: Dict[str, Any],
    entity_candidates_id: List[str],
    total_candidates: List[str],
    total_relations: List[str],
    total_entities_id: List[str],
    total_topic_entities: List[str],
    total_head: List[bool],
) -> Tuple[List, List, List, List, List]:
    """Update exploration history"""
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]

    candidates_relation = [ent_rel["relation"]] * len(entity_candidates)
    topic_entities = [ent_rel["entity"]] * len(entity_candidates)
    head_num = [ent_rel["head"]] * len(entity_candidates)

    total_candidates.extend(entity_candidates)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)

    return (
        total_candidates,
        total_relations,
        total_entities_id,
        total_topic_entities,
        total_head,
    )


def if_topic_non_retrieve(string: str) -> bool:
    """Check if entity should not be retrieved (e.g., numeric values)"""
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_all_digits(lst: List[str]) -> bool:
    """Check if all items in list are digits"""
    for s in lst:
        if not s.isdigit():
            return False
    return True


class MockFreebaseFunc:
    """Mock implementation for testing without Freebase deployment"""

    @staticmethod
    def entity_search(entity: str, relation: str, head: bool = True) -> List[str]:
        """Mock entity search - returns sample data"""
        mock_data = {
            "Taylor Swift": {
                "music.artist.album": ["1989", "Folklore", "Midnights"],
                "music.artist.track": ["Shake It Off", "Anti-Hero", "Blank Space"],
                "award.award_winner.awards_won": [
                    "Grammy Award",
                    "American Music Award",
                ],
            }
        }

        if entity in mock_data and relation in mock_data[entity]:
            return mock_data[entity][relation]
        return [f"mock_entity_{i}" for i in range(3)]

    @staticmethod
    def relation_search_prune(
        entity_id: str,
        sub_questions: List[str],
        entity_name: str,
        pre_relations: List[str],
        pre_head: bool,
        question: str,
        azure_config: Dict[str, Any],
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """Mock relation search"""
        mock_relations = [
            {"entity": entity_id, "relation": "music.artist.album", "head": True},
            {"entity": entity_id, "relation": "music.artist.track", "head": True},
            {
                "entity": entity_id,
                "relation": "award.award_winner.awards_won",
                "head": True,
            },
        ]
        return mock_relations, {"total": 100, "input": 50, "output": 50}


# Determine if we should use mock or real implementation
USE_MOCK = not os.path.exists(
    "/usr/local/bin/virtuoso-t"
)  # Check if Virtuoso is installed

if USE_MOCK:
    print(
        "⚠️ Using Mock Freebase implementation. Install Virtuoso and deploy Freebase for real KG queries."
    )
    # Override functions with mock versions
    entity_search = MockFreebaseFunc.entity_search
    relation_search_prune = MockFreebaseFunc.relation_search_prune
