import pytest
import ollama
from engine.retrieval.analysis_agent import generate_rag_response

# LLM-as-a-Judge Evaluation Logic
def judge_response(question, answer, context):
    prompt = f"""
    You are an impartial judge evaluating an AI assistant's response.
    Evaluate the response based on two criteria:
    1. Faithfulness: Is the answer strictly derived from the provided context? (Does it hallucinate?)
    2. Relevancy: Does the answer directly address the user's question?
    
    Context: {context}
    Question: {question}
    Answer: {answer}
    
    Output exactly ONE line in the following format:
    Faithfulness: [Yes/No], Relevancy: [Yes/No]
    """
    
    response = ollama.generate(model="gemma2:9b", prompt=prompt)
    verdict = response['response'].strip()
    return verdict

# PyTest Cases
@pytest.mark.parametrize("query", [
    "Are there any listings in Bondi?",
    "Find me a shared room for under $100."
])
def test_rag_system_accuracy(query):
    # 1. Generate RAG Response
    answer, context = generate_rag_response(query)
    
    # Ensure it didn't completely fail
    assert len(answer) > 5
    assert len(context) > 5
    
    # 2. Judge the Response
    verdict = judge_response(query, answer, context)
    print(f"\n[EVALUATION for '{query}']")
    print(f"Context Provided: {len(context)} characters")
    print(f"Answer: {answer}")
    print(f"Judge Verdict: {verdict}\n")
    
    # 3. Assert Passes
    # We expect Gemma2 to say "Faithfulness: Yes, Relevancy: Yes"
    assert "Faithfulness: Yes" in verdict, f"Failed Faithfulness: {verdict}"
    assert "Relevancy: Yes" in verdict, f"Failed Relevancy: {verdict}"
