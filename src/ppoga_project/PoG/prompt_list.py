# PoG Prompts (adapted from original PoG repository)

subobjective_prompt = """Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis.
Here is an example: 
Q: Which of the countries in the Caribbean has the smallest country calling code?
Output: ['Search the countries in the Caribbean', 'Search the country calling code for each Caribbean country', 'Compare the country calling codes to find the smallest one']

Now you need to directly output subobjectives of the following question in list format without other information or notes. 
Q: """

extract_relation_prompt = """Please provide as few highly relevant relations as possible to the question and its subobjectives from the following relations (separated by semicolons).
Here is an example:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
The output is: 
['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']

Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes.
Q: """

answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.

Here are examples:
Q: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., media_common.quotation.author, Thomas Jefferson
The output is:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said 'Taste cannot be controlled by law', which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson died."
}

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, book.written_work.author, Laura Ingalls Wilder
Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity
Unknown-Entity, people.place_lived.location, De Smet
The output is:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "De Smet"
    },
    "R": "Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet."
}

Now you need to directly output the results of the following question in JSON format (must include "A" and "R") without other information or notes.
Q: """

cot_prompt = """Please answer the question according to your knowledge step by step. Here is an example:
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
The output is:
{
    "A": {
        "Known": "Yes",
        "Answer": "Washington, D.C."
    },
    "R": "First, the education institution has a sports team named George Washington Colonials men's basketball is George Washington University. Second, George Washington University is in Washington D.C."
}

Please directly output the answer in JSON format (must include "A" and "R") without other information or notes.
Q: """
