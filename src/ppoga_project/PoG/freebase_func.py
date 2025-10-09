from SPARQLWrapper import SPARQLWrapper, JSON
import random
import json
import time
import re
import sys
import os
from .utils import run_llm

# SPARQL endpoint configuration
SPARQLPATH = "http://localhost:8890/sparql"  # Your local Freebase endpoint

# Pre-defined SPARQL queries
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}"""
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""


def abandon_rels(relation):
    """Filter out unnecessary relations"""
    if (
        relation == "type.object.type"
        or relation == "type.object.name"
        or relation.startswith("common.")
        or relation.startswith("freebase.")
        or "sameAs" in relation
    ):
        return True
    return False


def execute_sparql(sparql_query):
    """Execute SPARQL query against Freebase"""
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"SPARQL execution error: {e}")
        return []


def replace_relation_prefix(relations):
    """Remove Freebase namespace prefix from relations"""
    return [
        relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for relation in relations
    ]


def replace_entities_prefix(entities):
    """Remove Freebase namespace prefix from entities"""
    return [
        entity["tailEntity"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for entity in entities
    ]


def id2entity_name_or_type(entity_id):
    """Convert entity ID to human-readable name"""
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        if len(results["results"]["bindings"]) == 0:
            return entity_id
        else:
            return results["results"]["bindings"][0]["tailEntity"]["value"]
    except:
        return entity_id


def entity_search(entity, relation, head=True):
    """Search for entities connected via a specific relation"""
    if head:
        tail_entities_extract = sparql_tail_entities_extract % (entity, relation)
        entities = execute_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract % (relation, entity)
        entities = execute_sparql(head_entities_extract)

    entity_ids = replace_entities_prefix(entities)
    return entity_ids


def relation_search_prune(
    entity_id, sub_questions, entity_name, pre_relations, pre_head, question, args
):
    """Search and prune relations for an entity"""
    # Get head relations
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execute_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)

    # Get tail relations
    sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
    tail_relations = execute_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    # Filter unnecessary relations
    if args.remove_unnecessary_rel:
        head_relations = [
            relation for relation in head_relations if not abandon_rels(relation)
        ]
        tail_relations = [
            relation for relation in tail_relations if not abandon_rels(relation)
        ]

    # Remove previously explored relations
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()

    # Select relations using LLM (원본 PoG 방식 복원)
    if total_relations:
        # 관계 선택을 위한 프롬프트 구성
        prompt = f"""Please provide as few highly relevant relations as possible to the question and its subobjectives from the following relations (separated by semicolons).
Here is an example:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
The output is: 
['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']

Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes.
Q: {question}
Subobjectives: {sub_questions}
Topic Entity: {entity_name}
Relations: {'; '.join(total_relations)}"""

        try:
            # LLM 호출 (config.py의 필드명과 일치하도록 수정)
            result, token_info = run_llm(
                prompt=prompt,
                temperature=0.3,
                max_tokens=4096,
                openai_api_keys=getattr(
                    args, "openai_api_keys", ""
                ),  # config의 openai_api_keys 필드 사용 (복수형)
                engine=getattr(
                    args, "LLM_type", "gpt-3.5-turbo"
                ),  # config의 LLM_type 필드 사용
                print_in=False,
                print_out=False,
            )

            print(f"🔍 LLM 관계 선택 결과: {result[:100]}...")

            # 원본 PoG의 select_relations 로직 적용
            selected_relations = []
            try:
                # 대괄호 안의 내용 추출
                last_brace_l = result.rfind("[")
                last_brace_r = result.rfind("]")

                if last_brace_l < last_brace_r:
                    result = result[last_brace_l : last_brace_r + 1]

                rel_list = eval(result.strip())
                selected_relations = [rel for rel in rel_list if rel in total_relations]
                print(
                    f"✅ LLM이 선택한 관계 {len(selected_relations)}개: {selected_relations[:3]}..."
                )

            except Exception as parse_error:
                # 파싱 실패 시 처음 5개 관계 사용 (fallback)
                print(f"❌ LLM 응답 파싱 실패: {parse_error}, fallback 사용")
                selected_relations = total_relations[:5]

        except Exception as e:
            print(f"❌ LLM 관계 선택 실패: {e}, fallback 사용")
            selected_relations = total_relations[:5]
            token_info = {"total": 0, "input": 0, "output": 0}

        relations = []
        for relation in selected_relations:
            if relation in head_relations:
                relations.append(
                    {"entity": entity_id, "relation": relation, "head": True}
                )
            elif relation in tail_relations:
                relations.append(
                    {"entity": entity_id, "relation": relation, "head": False}
                )

        print(f"🎯 최종 선택된 관계 {len(relations)}개")
        return relations, token_info
    else:
        return [], {"total": 0, "input": 0, "output": 0}


def provide_triple(entity_candidates_id, relation):
    """Convert entity IDs to names and return sorted lists"""
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
