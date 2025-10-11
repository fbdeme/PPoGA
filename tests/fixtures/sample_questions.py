"""
Sample questions for testing PPoGA system
"""

# Simple single-hop questions
SIMPLE_QUESTIONS = [
    {
        "question": "Who directed The Godfather?",
        "expected_answer": "Francis Ford Coppola",
        "complexity": "simple",
        "hops": 1,
        "entities": ["The Godfather"],
        "relations": ["film.film.directed_by"],
    },
    {
        "question": "What is the capital of France?",
        "expected_answer": "Paris",
        "complexity": "simple",
        "hops": 1,
        "entities": ["France"],
        "relations": ["location.country.capital"],
    },
    {
        "question": "When was Barack Obama born?",
        "expected_answer": "August 4, 1961",
        "complexity": "simple",
        "hops": 1,
        "entities": ["Barack Obama"],
        "relations": ["people.person.date_of_birth"],
    },
]

# Complex multi-hop questions
COMPLEX_QUESTIONS = [
    {
        "question": "Who is the spouse of the director of The Godfather?",
        "expected_answer": "Eleanor Coppola",
        "complexity": "complex",
        "hops": 3,
        "entities": ["The Godfather", "Francis Ford Coppola", "Eleanor Coppola"],
        "relations": [
            "film.film.directed_by",
            "people.person.spouse_s",
            "people.marriage.spouse",
        ],
    },
    {
        "question": "What movies did the spouse of Steven Spielberg direct?",
        "expected_answer": ["The Love Letter", "Unstrung Heroes"],
        "complexity": "complex",
        "hops": 4,
        "entities": ["Steven Spielberg", "Kate Capshaw"],
        "relations": [
            "people.person.spouse_s",
            "people.marriage.spouse",
            "film.director.film",
        ],
    },
]

# Benchmark questions for performance testing
BENCHMARK_QUESTIONS = [
    {
        "question": "What are all the movies directed by Christopher Nolan?",
        "complexity": "medium",
        "expected_count": 12,
        "test_type": "list_retrieval",
    },
    {
        "question": "Who are the actors in all Christopher Nolan movies?",
        "complexity": "high",
        "expected_count": 50,
        "test_type": "complex_aggregation",
    },
]

# All questions combined
ALL_TEST_QUESTIONS = SIMPLE_QUESTIONS + COMPLEX_QUESTIONS + BENCHMARK_QUESTIONS
