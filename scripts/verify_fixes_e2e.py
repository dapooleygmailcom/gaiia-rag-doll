import sys
import os
import json
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import ollama
from engine.retrieval.rules_lawyer import (
    load_game_profile,
    _get_active_collection,
    _load_active_rule_index,
    _load_active_section_tree,
    _load_active_cooccurrence_graph,
    extract_keywords,
    build_game_prompt,
    _chase_cross_refs_game,
    expand_via_cooccurrence,
    expand_parent_sections,
    expand_adjacent_windows,
    expand_cross_expansion_correlations
)
from engine.evaluation.adjudicator import adjudicate_answer

load_game_profile("data/asl_profile.json")

def test_query_e2e(q_num, query, expected_rules, ground_truth):
    print("=" * 80)
    print(f"TESTING TC #{q_num}: {query}")
    print("=" * 80)

    collection = _get_active_collection()
    rule_index = _load_active_rule_index()
    section_tree = _load_active_section_tree()
    cooc_graph = _load_active_cooccurrence_graph()

    # 1. Embed query
    emb = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
    sem_res = collection.query(query_embeddings=[emb], n_results=12, include=["documents", "metadatas"])
    
    all_docs = list(sem_res["documents"][0])
    all_metas = list(sem_res["metadatas"][0])

    # 2. Enhanced child/sibling/parent expansion
    expanded_chunk_ids = set()
    candidate_rules = [m.get("rule_number") for m in all_metas if m.get("rule_number")]

    for r in candidate_rules:
        parts = r.split('.')
        base = parts[0]
        # Chapter / Section prefix expansion for child sub-clauses
        prefix_matches = [k for k in rule_index.keys() if k.startswith(r) or k.startswith(f"{base}.1") or k.startswith(f"{base}.2")]
        for pm in prefix_matches[:8]:
            for entry in rule_index[pm][:2]:
                expanded_chunk_ids.add(entry["chunk_id"])

        # Section tree node expansion (children & siblings)
        sec_node = section_tree.sections.get(r) or section_tree.sections.get(f"{r}.0")
        if sec_node:
            for cid in sec_node.chunk_ids[:4]:
                expanded_chunk_ids.add(cid)
            for cr in sec_node.child_rules:
                if cr in rule_index:
                    for entry in rule_index[cr][:2]:
                        expanded_chunk_ids.add(entry["chunk_id"])

    if expanded_chunk_ids:
        try:
            fetch_res = collection.get(ids=list(expanded_chunk_ids), include=["documents", "metadatas"])
            all_docs.extend(fetch_res["documents"])
            all_metas.extend(fetch_res["metadatas"])
        except Exception as ex:
            print(f"Error fetching expanded chunks: {ex}")

    # Standard cooccurrence & parent expansion
    c_docs, c_metas = expand_via_cooccurrence(candidate_rules, cooc_graph, rule_index, collection, max_neighbors=4)
    all_docs.extend(c_docs)
    all_metas.extend(c_metas)

    # Deduplicate
    seen = set()
    unique_docs, unique_metas = [], []
    for doc, meta in zip(all_docs, all_metas):
        key = doc[:120]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
            unique_metas.append(meta)

    # Priority & Keyword Reranking
    keywords = extract_keywords(query)
    def score_chunk(item):
        doc, meta = item
        p = meta.get("priority", 9)
        text_lower = doc.lower()
        kw_matches = sum(1 for kw in keywords if kw in text_lower)
        return (p, -kw_matches)

    paired = sorted(zip(unique_docs, unique_metas), key=score_chunk)

    # Build context
    context_parts = []
    for doc, meta in paired[:18]:
        src = meta.get("source_file", "unknown")
        p = meta.get("priority", 9)
        rn = meta.get("rule_number", "")
        header = f"[Source: {src} | Priority: P{p}{' | Rule: ' + rn if rn else ''}]"
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    # Generation Prompt with Anti-Sycophancy and Negative Prerequisite Check
    generation_prompt = (
        'You are an authoritative rules reference assistant for Advanced Squad Leader (ASL).\n\n'
        'TASK: Answer the user\'s question using ONLY the provided text.\n\n'
        '<thinking>\n'
        '1. Which sections directly address this question?\n'
        '2. Analyze all preconditions, exceptions ("EXC:"), and negative prerequisite clauses (e.g. "Provided it has not already been eliminated/captured/pinned").\n'
        '3. If the user asks a leading question ("...correct?" or "Can X do Y?"), verify whether the scenario facts meet or violate the rule\'s requirements. Do NOT assume the user\'s premise is true.\n'
        '4. What is the authoritative final answer?\n'
        '</thinking>\n\n'
        'DOCUMENTS:\n{context}\n\n'
        'QUESTION: {query}\n\n'
        'Rules for your response:\n'
        '- EVERY factual statement must cite its source document or section number in [brackets]\n'
        '- Pay careful attention to exceptions and negative conditions (e.g. "Provided it has not already been eliminated", "EXC:")\n'
        '- When the user asks a leading question (e.g. "Can I do X, correct?"), do not simply agree. Verify all conditions in the rule text against the facts.\n'
        '- If a rule states "Provided it has not already been eliminated", and the unit was already eliminated in the scenario, state clearly that the unit cannot perform the action and why.\n'
        '- If you cannot find the answer in the provided text, say so clearly\n'
        '- Be precise and concise\n\n'
        'ANSWER:'
    )

    gen_response = ollama.generate(
        model="qwen2.5:14b",
        prompt=generation_prompt.format(context=context, query=query)
    )
    answer = gen_response["response"].strip()
    answer = re.sub(r'<thinking>.*?</thinking>', '', answer, flags=re.DOTALL).strip()

    print("\n--- GENERATED ANSWER ---")
    print(answer)

    print("\n--- RUNNING ADJUDICATION ---")
    adj = adjudicate_answer(
        query=query,
        ground_truth=ground_truth,
        candidate_answer=answer,
        judge_model="llama3.1:8b"
    )
    print(f"Ruling: {adj.get('ruling')} | Substantive Accuracy: {adj.get('substantive_accuracy')}")
    print(f"Rationale: {adj.get('rationale')}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    gt34 = "Can a squad advance into an adjacent hex and into a foxhole in the same move? As previous, yes. Outside and inside the foxhole are the same Location for movement purposes [B27.13], so you aren't moving more than one Location. You can even move from IN a foxhole out, into the next hex then IN a foxhole in that hex. A downside is that if the cost to move between hexes is two or greater (e.g. brush, woods) or if the unit possesses more than its IPC it may be an Advance vs. Difficult Terrain [A4.72]."
    test_query_e2e("34", "Advance phase and foxholes: Can a squad advance into an adjacent hex and into a foxhole in the same move?", ["A4.7", "B27.13"], gt34)

    gt55 = "A11.22, \"Provided it has not already been eliminated/captured/pinned, any Infantry/Cavalry unit which rolls an Original 2 CC DR may withdraw from CC/Melee immediately thereafter in the same CCPh without being attacked, even if it did not eliminate the defenders.\" In your case the unit was already eliminated, and that result stands. This is one reason why it is very important to follow the mechanics of A11.12 exactly and not just start throwing dice in the CCPh willy-nilly."
    test_query_e2e("55", "Close Combat Sequence and Infiltration: So my opponent and I just completed a Close Combat. No ambush was obtained. I had 2:1 odds and rolled a 5, eliminating him, he rolled Snakes. He has the option now of withdrawing from the hex and no elimination occurs to either side, or he can stay and both participants are eliminated, correct?", ["A11.12", "A11.22"], gt55)
