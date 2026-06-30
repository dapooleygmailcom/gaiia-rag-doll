import pytest
from engine.retrieval.analysis_agent import generate_rag_response
from test_rag_accuracy import judge_response

# Define the TDD Scenarios
TDD_SCENARIOS = [
    {
        "type": "Fine-Grained",
        "query": "A quiet private room under $80 with a rating of at least 4.8.",
        "expected_behavior": "Strict metadata extraction (price_max=80, rating_min=4.8). Should return only matching rooms or a faithful denial."
    },
    {
        "type": "Loose Semantic",
        "query": "I want a luxury villa with a pool and ocean views.",
        "expected_behavior": "No metadata filters triggered. Relies purely on ChromaDB vector distance for semantic match."
    },
    {
        "type": "Boundary Edge Case",
        "query": "Show me a place for exactly $0.",
        "expected_behavior": "Metadata filter price_max=0. Should faithfully return 'I could not find any matching listings'."
    },
    {
        "type": "Out of Domain / Negative",
        "query": "What is the capital of France?",
        "expected_behavior": "ChromaDB returns irrelevant listings. LLM truthfully rejects the question based on context."
    }
]

@pytest.mark.parametrize("scenario", TDD_SCENARIOS, ids=lambda s: s["type"])
def test_rag_tdd_scenarios(scenario):
    query = scenario["query"]
    
    print(f"\n{'='*50}")
    print(f"SCENARIO TYPE: {scenario['type']}")
    print(f"QUERY: {query}")
    print(f"EXPECTED: {scenario['expected_behavior']}")
    print(f"{'='*50}")
    
    # 1. Generate RAG Response
    answer, context = generate_rag_response(query)
    
    # 2. Judge the Response
    verdict = judge_response(query, answer, context)
    
    print(f"\n[ACTUAL OUTCOME]")
    print(f"Retrieved Context length: {len(context)} chars")
    print(f"Agent Answer: {answer}")
    print(f"Judge Verdict: {verdict}")
    
    # We assert that the LLM was Faithful to the retrieved context, regardless of relevancy
    assert "Faithfulness: Yes" in verdict, f"Model Hallucinated! Verdict: {verdict}"
    
    # We do NOT rigidly assert Relevancy here because the out-of-domain edge case
    # might correctly result in 'Relevancy: No' or 'Relevancy: N/A' if the LLM refuses to answer.
    # The goal of this TDD script is to surface the outcomes to stdout.
