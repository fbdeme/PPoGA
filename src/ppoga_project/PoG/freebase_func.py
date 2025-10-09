from SPARQLWrapper import SPARQLWrapper, JSON
import random
import json
import time
import re

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
    
    # Prioritize important relations (film, people, etc.)
    def prioritize_relations(relations):
        priority_patterns = [
            'film.film.directed_by',
            'film.film.starring',
            'film.film.produced_by',
            'film.film.written_by',
            'people.person',
            'film.',
            'music.',
            'book.',
            'organization.'
        ]
        
        prioritized = []
        regular = []
        
        for relation in relations:
            is_priority = False
            for pattern in priority_patterns:
                if pattern in relation:
                    prioritized.append(relation)
                    is_priority = True
                    break
            if not is_priority:
                regular.append(relation)
        
        return sorted(prioritized) + sorted(regular)
    
    total_relations = prioritize_relations(total_relations)

    # Select relations using LLM (simplified version)
    if total_relations:
        # For now, return first few relations (can be enhanced with LLM selection)
        selected_relations = total_relations[:5]  # Limit to 5 relations

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

        return relations, {"total": 0, "input": 0, "output": 0}
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
