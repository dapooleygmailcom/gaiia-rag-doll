import json
import time
import os
import ollama
from engine.retrieval.policy_agent import compare_policies

# Ensure log directory exists
os.makedirs("data/logs", exist_ok=True)
LOG_FILE = "data/logs/policy_test_results.json"

POLICY_TEST_CASES = [
    {
        "id": 1,
        "query": "Compare flood damage coverage and policy exclusions across AAMI, CBA, and TIO.",
        "carriers": ["AAMI", "CBA", "TIO"]
    },
    {
        "id": 2,
        "query": "What is the maximum limit for jewelry and watches under AAMI vs CBA vs Allianz_NAB?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB"]
    },
    {
        "id": 3,
        "query": "Which policies offer coverage for guest or visitor belongings, and what are their limits?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ"]
    },
    {
        "id": 4,
        "query": "Compare the coverage and exclusions for accidental glass breakage across AAMI, CBA, and TIO.",
        "carriers": ["AAMI", "CBA", "TIO"]
    },
    {
        "id": 5,
        "query": "What is the policy limit for portable valuables (like mobile phones) away from home under AAMI, CBA, and CGU_ANZ?",
        "carriers": ["AAMI", "CBA", "CGU_ANZ"]
    },
    {
        "id": 6,
        "query": "Which policies cover damage caused by domestic pets, and what are the exclusions?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "Westpac_FSR"]
    },
    {
        "id": 7,
        "query": "Compare new-for-old replacement terms under Allianz_NAB and Westpac_FSR.",
        "carriers": ["Allianz_NAB", "Westpac_FSR"]
    },
    {
        "id": 8,
        "query": "What are the standard excesses and voluntary excess discounts across AAMI, CBA, and TIO?",
        "carriers": ["AAMI", "CBA", "TIO"]
    },
    {
        "id": 9,
        "query": "Compare the coverage for temporary accommodation if the home is uninhabitable across AAMI, CBA, and Allianz_NAB.",
        "carriers": ["AAMI", "CBA", "Allianz_NAB"]
    },
    {
        "id": 10,
        "query": "Which policy has the most comprehensive cover for outdoor items like plants and garden furniture?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    # Additional 15 Scenarios
    {
        "id": 11,
        "query": "Compare the waiting periods (e.g. 48/72 hours) for bushfire, storm surge, and cyclone coverage across Westpac_FSR, CBA, and TIO.",
        "carriers": ["Westpac_FSR", "CBA", "TIO"]
    },
    {
        "id": 12,
        "query": "Which policies cover damage caused by storm surge, action of the sea, or tsunamis? What are the specific exclusions?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    {
        "id": 13,
        "query": "Compare landslide, mudslide, and subsidence coverage. How many hours after a storm or flood must the landslide occur to be covered under AAMI, CBA, and TIO?",
        "carriers": ["AAMI", "CBA", "TIO"]
    },
    {
        "id": 14,
        "query": "What are the standard contents limits for non-motorised bicycles and e-bikes under CBA vs Westpac_FSR vs Allianz_NAB, and is there an option to increase them?",
        "carriers": ["CBA", "Westpac_FSR", "Allianz_NAB"]
    },
    {
        "id": 15,
        "query": "Compare the coverage limits, fixed caps, and exclusions for tools of trade, commercial stock, and home-office business equipment kept at the home address.",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "Westpac_FSR"]
    },
    {
        "id": 16,
        "query": "What are the total group limits and per-item limits for stamp, coin, medal, or card collections under AAMI, CBA, and Allianz_NAB?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB"]
    },
    {
        "id": 17,
        "query": "Is accidental damage cover (e.g., dropping a phone or spilling wine on carpet) automatically included in the basic policy, or is it an optional extra across all 8 carriers?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    {
        "id": 18,
        "query": "If a tenant is legally liable under a lease, what building damage (like fixed glass or toilet cisterns) is covered under their contents-only policy under AAMI, TIO, and CBA?",
        "carriers": ["AAMI", "TIO", "CBA"]
    },
    {
        "id": 19,
        "query": "Do any of the policies cover loss, theft, or malicious damage caused by tenants, paying guests (like Airbnb), or boarders?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    {
        "id": 20,
        "query": "What are the age limits or exclusions for outdoor shade sails, canvas, and awnings under CBA, Allianz_NAB, and AAMI?",
        "carriers": ["CBA", "Allianz_NAB", "AAMI"]
    },
    {
        "id": 21,
        "query": "Compare how policies cover or exclude water damage caused by gradual water seepage, slow leaks, rising damp, or mold.",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    {
        "id": 22,
        "query": "What are the policy exclusions regarding structural defects, faulty design, poor workmanship, or use of substandard materials?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "Westpac_FSR"]
    },
    {
        "id": 23,
        "query": "What is the maximum public/legal liability coverage cap across all policies, and does it extend to cover incidents occurring outside Australia?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    {
        "id": 24,
        "query": "How many continuous unoccupied days (e.g., 60 days) are allowed before the home building/contents cover is suspended or subject to an additional excess?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    {
        "id": 25,
        "query": "Do any of the standard exclusions include cyber-attack losses, war, or radioactive contamination across the carriers?",
        "carriers": ["AAMI", "CBA", "Allianz_NAB", "TIO", "CGU_ANZ", "Westpac_FSR"]
    },
    # Specialized Reference & Binding UATs
    {
        "id": 26,
        "query": "Under Allianz_NAB contents cover, does the alternative accommodation benefit pay out if I am already claiming it under my buildings cover? Trace the reference.",
        "carriers": ["Allianz_NAB"]
    },
    {
        "id": 27,
        "query": "What is excluded under CBA regarding loose surfaces of paths and driveways?",
        "carriers": ["CBA"]
    }
]

def judge_response(question, answer, context):
    prompt = f"""
    You are an impartial judge evaluating an AI assistant's response in a policy comparison RAG system.
    Evaluate the response based on two criteria:
    1. Faithfulness: Is the answer strictly derived from the provided context? (Does it hallucinate?)
    2. Relevancy: Does the answer directly address the user's comparison question?
    
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
    print("STARTING GAIIA ADVANCED POLICY COMPARISON UAT (27 SCENARIOS)")
    print("=" * 60)
    
    results = []
    total_faithful = 0
    total_relevant = 0
    total_perfect = 0
    
    start_time = time.time()
    total_cases = len(POLICY_TEST_CASES)
    
    for idx, tc in enumerate(POLICY_TEST_CASES, 1):
        query = tc["query"]
        carriers = tc["carriers"]
        tc_id = tc["id"]
        
        print(f"\n--- [Case {idx}/{total_cases}] ID: {tc_id} ---")
        print(f"Query: {query}")
        print(f"Carriers: {', '.join(carriers)}")
        
        case_start = time.time()
        
        try:
            answer, context = compare_policies(query, carriers=carriers)
        except Exception as e:
            print(f"RAG comparison execution failed: {e}")
            answer, context = "ERROR", "ERROR"
            
        verdict = judge_response(query, answer, context)
        faithful, relevant = parse_verdict(verdict)
        
        case_duration = time.time() - case_start
        print(f"Time Taken: {case_duration:.1f}s")
        print(f"Verdict: {verdict}")
        print(f"Parsed -> Faithfulness: {faithful} | Relevancy: {relevant}")
        
        is_faithful = (faithful == "Yes")
        is_relevant = (relevant == "Yes")
        
        if is_faithful:
            total_faithful += 1
        if is_relevant:
            total_relevant += 1
        if is_faithful and is_relevant:
            total_perfect += 1
            
        results.append({
            "id": tc_id,
            "query": query,
            "carriers": carriers,
            "answer": answer,
            "context": context,
            "verdict": verdict,
            "faithfulness": faithful,
            "relevancy": relevant,
            "duration_seconds": round(case_duration, 2)
        })
        
        # Intermediate backup to file
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
    total_duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("ALL POLICY TESTS COMPLETED")
    print(f"Total Time Taken: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print("=" * 60)
    
    # Print markdown table to stdout
    print("\n### Policy Comparison RAG Accuracy Table\n")
    print("| Metric | Total Cases | Passes | Pass Rate (%) |")
    print("| --- | --- | --- | --- |")
    
    faith_rate = (total_faithful / total_cases * 100)
    rel_rate = (total_relevant / total_cases * 100)
    perf_rate = (total_perfect / total_cases * 100)
    
    print(f"| Faithfulness (No Hallucinations) | {total_cases} | {total_faithful} | {faith_rate:.1f}% |")
    print(f"| Relevancy (Answers the Question) | {total_cases} | {total_relevant} | {rel_rate:.1f}% |")
    print(f"| **Perfect UAT Pass Rate** | **{total_cases}** | **{total_perfect}** | **{perf_rate:.1f}%** |")
    
    # Save statistics metadata block
    summary_data = {
        "total_duration_minutes": round(total_duration / 60, 2),
        "total_cases": total_cases,
        "total_faithful": total_faithful,
        "total_relevant": total_relevant,
        "total_perfect": total_perfect,
        "perfect_pass_rate": round(perf_rate, 2)
    }
    
    with open("data/logs/policy_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    print("\nSaved detailed UAT logs to data/logs/policy_test_results.json and summary to data/logs/policy_test_summary.json.")

if __name__ == "__main__":
    main()
