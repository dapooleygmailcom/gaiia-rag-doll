import json
import time
import sys
import os
import ollama
from engine.retrieval.analysis_agent import generate_rag_response

# Ensure the log folder exists
os.makedirs("data/logs", exist_ok=True)
LOG_FILE = "data/logs/massive_test_results.json"

TEST_CASES = [
    # Category 1: Fine-Grained (Strict Metadata + Semantic) - 15 cases
    {"id": 1, "category": "Fine-Grained", "query": "A private room in Surry Hills under $80 with a rating of at least 4.8."},
    {"id": 2, "category": "Fine-Grained", "query": "Shared room in Parramatta under $60 with a rating of at least 3.5."},
    {"id": 3, "category": "Fine-Grained", "query": "Entire home/apt in Bondi under $200 with a rating of at least 4.5."},
    {"id": 4, "category": "Fine-Grained", "query": "A private room in Newtown under $100 with a rating of at least 4.0."},
    {"id": 5, "category": "Fine-Grained", "query": "Shared room in Manly under $70 with a rating of at least 4.2."},
    {"id": 6, "category": "Fine-Grained", "query": "Entire home/apt in Darlinghurst under $180 with a rating of at least 4.6."},
    {"id": 7, "category": "Fine-Grained", "query": "A private room in Bondi under $120 with a rating of at least 4.5."},
    {"id": 8, "category": "Fine-Grained", "query": "Entire home/apt in Surry Hills under $250 with a rating of at least 4.7."},
    {"id": 9, "category": "Fine-Grained", "query": "Shared room in Newtown under $50 with a rating of at least 3.0."},
    {"id": 10, "category": "Fine-Grained", "query": "A private room in Parramatta under $90 with a rating of at least 4.1."},
    {"id": 11, "category": "Fine-Grained", "query": "Entire home/apt in Manly under $300 with a rating of at least 4.9."},
    {"id": 12, "category": "Fine-Grained", "query": "Shared room in Darlinghurst under $80 with a rating of at least 3.8."},
    {"id": 13, "category": "Fine-Grained", "query": "A private room in Manly under $110 with a rating of at least 4.4."},
    {"id": 14, "category": "Fine-Grained", "query": "Entire home/apt in Newtown under $160 with a rating of at least 4.3."},
    {"id": 15, "category": "Fine-Grained", "query": "Shared room in Surry Hills under $75 with a rating of at least 4.0."},

    # Category 2: Loose Semantic (Semantic Search only) - 15 cases
    {"id": 16, "category": "Loose Semantic", "query": "I want a luxury villa with a pool and ocean views."},
    {"id": 17, "category": "Loose Semantic", "query": "Find a quiet and peaceful place for a romantic getaway."},
    {"id": 18, "category": "Loose Semantic", "query": "A very noisy and bustling room close to the nightlife."},
    {"id": 19, "category": "Loose Semantic", "query": "Extremely clean and modern apartment with high-speed wifi."},
    {"id": 20, "category": "Loose Semantic", "query": "Something outdated but very affordable and cozy."},
    {"id": 21, "category": "Loose Semantic", "query": "Basic room, clean but tiny with a friendly host."},
    {"id": 22, "category": "Loose Semantic", "query": "A terrible experience where the place was dirty and loud."},
    {"id": 23, "category": "Loose Semantic", "query": "A lovely, quiet neighborhood perfect for families close to the beach."},
    {"id": 24, "category": "Loose Semantic", "query": "A modern apartment with super fast internet access."},
    {"id": 25, "category": "Loose Semantic", "query": "A place with a pool, great ocean scenery, and a premium experience."},
    {"id": 26, "category": "Loose Semantic", "query": "A quiet getaway for a couple seeking peace."},
    {"id": 27, "category": "Loose Semantic", "query": "Cozy, simple room that doesn't cost much."},
    {"id": 28, "category": "Loose Semantic", "query": "Active nightlife location with lots of noise and bars nearby."},
    {"id": 29, "category": "Loose Semantic", "query": "Clean room where the owner is nice and welcoming."},
    {"id": 30, "category": "Loose Semantic", "query": "An unhygienic and loud lodging that had a bad rating."},

    # Category 3: Boundary / Edge Cases (Min/Max values or extreme conditions) - 10 cases
    {"id": 31, "category": "Boundary/Edge", "query": "Show me a place for exactly $0."},
    {"id": 32, "category": "Boundary/Edge", "query": "Find me a listing under $1."},
    {"id": 33, "category": "Boundary/Edge", "query": "A private room with a rating of at least 6.0."},
    {"id": 34, "category": "Boundary/Edge", "query": "A shared room in Surry Hills under $10."},
    {"id": 35, "category": "Boundary/Edge", "query": "Entire home/apt under $5 with rating 5.0."},
    {"id": 36, "category": "Boundary/Edge", "query": "A room in Mars under $100."},
    {"id": 37, "category": "Boundary/Edge", "query": "Show me listings with a negative price like -$50."},
    {"id": 38, "category": "Boundary/Edge", "query": "A castle in Surry Hills with a private helipad."},
    {"id": 39, "category": "Boundary/Edge", "query": "A room that has a rating of at least 0.0."},
    {"id": 40, "category": "Boundary/Edge", "query": "Entire home/apt under $999999 with rating 1.0."},

    # Category 4: Out of Domain / Negatives (Irrelevant to Airbnb) - 10 cases
    {"id": 41, "category": "Out of Domain", "query": "What is the capital of France?"},
    {"id": 42, "category": "Out of Domain", "query": "How do you write a quicksort algorithm in Python?"},
    {"id": 43, "category": "Out of Domain", "query": "Who won the FIFA World Cup in 2022?"},
    {"id": 44, "category": "Out of Domain", "query": "Explain the theory of relativity in simple terms."},
    {"id": 45, "category": "Out of Domain", "query": "What is 2 + 2 * 5?"},
    {"id": 46, "category": "Out of Domain", "query": "Tell me a joke about programming."},
    {"id": 47, "category": "Out of Domain", "query": "What is the main chemical component of water?"},
    {"id": 48, "category": "Out of Domain", "query": "Who wrote the play Hamlet?"},
    {"id": 49, "category": "Out of Domain", "query": "How far is the Moon from the Earth?"},
    {"id": 50, "category": "Out of Domain", "query": "What is the capital of Australia?"}
]

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
    try:
        response = ollama.generate(model="gemma2:9b", prompt=prompt)
        verdict = response['response'].strip()
        return verdict
    except Exception as e:
        print(f"Error calling Ollama judge: {e}")
        return "Faithfulness: Error, Relevancy: Error"

def parse_verdict(verdict_text):
    verdict_lower = verdict_text.lower()
    
    # Defaults
    faithfulness = "No"
    relevancy = "No"
    
    if "faithfulness: yes" in verdict_lower:
        faithfulness = "Yes"
    elif "faithfulness: no" in verdict_lower:
        faithfulness = "No"
        
    if "relevancy: yes" in verdict_lower:
        relevancy = "Yes"
    elif "relevancy: no" in verdict_lower:
        relevancy = "No"
        
    return faithfulness, relevancy

def main():
    print("=" * 60)
    print("STARTING GAIIA MASSIVE RAG TDD SUITE (50 SCENARIOS)")
    print("=" * 60)
    
    results = []
    
    # Track statistics per category
    # categories: Fine-Grained, Loose Semantic, Boundary/Edge, Out of Domain
    stats = {
        "Fine-Grained": {"total": 0, "faithful": 0, "relevant": 0, "both": 0},
        "Loose Semantic": {"total": 0, "faithful": 0, "relevant": 0, "both": 0},
        "Boundary/Edge": {"total": 0, "faithful": 0, "relevant": 0, "both": 0},
        "Out of Domain": {"total": 0, "faithful": 0, "relevant": 0, "both": 0}
    }
    
    start_time = time.time()
    
    for idx, tc in enumerate(TEST_CASES, 1):
        cat = tc["category"]
        query = tc["query"]
        tc_id = tc["id"]
        
        print(f"\n--- [Case {idx}/50] ID: {tc_id} | Category: {cat} ---")
        print(f"Query: {query}")
        
        case_start = time.time()
        
        # 1. Run through the hybrid RAG agent
        try:
            answer, context = generate_rag_response(query)
        except Exception as e:
            print(f"RAG Generation failed: {e}")
            answer, context = "ERROR", "ERROR"
            
        # 2. Score with LLM Judge
        verdict = judge_response(query, answer, context)
        faithful, relevant = parse_verdict(verdict)
        
        case_duration = time.time() - case_start
        print(f"Time Taken: {case_duration:.1f}s")
        print(f"Answer: {answer[:120]}..." if len(answer) > 120 else f"Answer: {answer}")
        print(f"Verdict: {verdict}")
        print(f"Parsed -> Faithfulness: {faithful} | Relevancy: {relevant}")
        
        # Update stats
        stats[cat]["total"] += 1
        is_faithful = (faithful == "Yes")
        is_relevant = (relevant == "Yes")
        
        if is_faithful:
            stats[cat]["faithful"] += 1
        if is_relevant:
            stats[cat]["relevant"] += 1
        if is_faithful and is_relevant:
            stats[cat]["both"] += 1
            
        # Save result object
        results.append({
            "id": tc_id,
            "category": cat,
            "query": query,
            "answer": answer,
            "context": context,
            "verdict": verdict,
            "faithfulness": faithful,
            "relevancy": relevant,
            "duration_seconds": round(case_duration, 2)
        })
        
        # Intermediate backup to file in case of crash
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
    total_duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print(f"Total Time Taken: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print("=" * 60)
    
    # Print markdown table to stdout
    print("\n### RAG Confidence and Accuracy Table\n")
    print("| Category | Total Cases | Faithfulness Pass | Relevancy Pass | Perfect Pass Rate (%) |")
    print("| --- | --- | --- | --- | --- |")
    
    overall_total = 0
    overall_faithful = 0
    overall_relevant = 0
    overall_both = 0
    
    for cat, data in stats.items():
        total = data["total"]
        faithful = data["faithful"]
        relevant = data["relevant"]
        both = data["both"]
        
        overall_total += total
        overall_faithful += faithful
        overall_relevant += relevant
        overall_both += both
        
        pass_rate = (both / total * 100) if total > 0 else 0
        print(f"| {cat} | {total} | {faithful} | {relevant} | {pass_rate:.1f}% |")
        
    overall_pass_rate = (overall_both / overall_total * 100) if overall_total > 0 else 0
    print(f"| **OVERALL** | **{overall_total}** | **{overall_faithful}** | **{overall_relevant}** | **{overall_pass_rate:.1f}%** |")
    
    # Save statistics metadata block
    summary_data = {
        "total_duration_minutes": round(total_duration / 60, 2),
        "overall_total": overall_total,
        "overall_faithful": overall_faithful,
        "overall_relevant": overall_relevant,
        "overall_both": overall_both,
        "overall_pass_rate": round(overall_pass_rate, 2),
        "category_stats": stats
    }
    
    with open("data/logs/massive_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    print("\nSaved detailed log to data/logs/massive_test_results.json and summary to data/logs/massive_test_summary.json.")

if __name__ == "__main__":
    main()
