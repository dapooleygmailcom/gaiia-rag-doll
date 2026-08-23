import os
import re
import json
import time
import html
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
}

RULE_REGEX = re.compile(r'(?:^|\s|[\[\(\{\:,])(\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)(?:[\]\)\}\:\,\.\s]|$)', re.MULTILINE)
SCENARIO_REGEX = re.compile(r'\b(?:[Ss]cenario|[Ss]cenarios)\s+([A-Z0-9]+)\b')

CORE_CONCEPTS = {
    "relative_range": ["relative range", "range chit", "rr5", "rr4", "rr3", "rr2", "rr1", "rr 5", "range determination"],
    "movement": ["movement", "lateral", "flank", "group transfer", "rush", "withdraw", "advance", "moving fire"],
    "fire_combat": ["fire card", "firepower", "modified fp", "to hit", "cover", "column", "defense strength", "dr", "drm"],
    "pinning_morale": ["pinned", "pin", "break", "rally", "morale", "cower", "coward", "panic"],
    "terrain": ["terrain", "open ground", "woods", "marsh", "brush", "building", "hill", "stream", "wall", "wire", "minefield"],
    "smoke_concealment": ["smoke", "conceal", "concealment", "hidden", "spotted", "sniper"],
    "melee_infiltration": ["melee", "hand to hand", "close combat", "infiltrate", "infiltration", "capture", "prisoner"],
    "vehicles_afv": ["vehicle", "afv", "tank", "panzer", "sherman", "t-34", "gun", "anti-tank", "armor", "hull", "turret"],
    "weapons_ordnance": ["machine gun", "mg", "lmg", "hmg", "mortar", "bazooka", "panzerfaust", "grenade", "flamethrower"],
    "heroes_leaders": ["leader", "hero", "sl", "asl", "nco", "officer", "command"],
    "cards_hand": ["draw deck", "discard", "hand size", "pass", "action card", "deck depletion", "reshuffle"]
}

def clean_text(raw_text):
    if not raw_text:
        return ""
    text = html.unescape(raw_text)
    # Convert quote blocks nicely
    text = re.sub(r'\[quote(?:=[^\]]*)?\](.*?)\[/quote\]', r'\n> \1\n', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip BBCode tags
    text = re.sub(r'\[/?[a-zA-Z0-9_-]+(?:=[^\]]*)?\]', '', text)
    # Strip HTML tags
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?[a-zA-Z0-9_-]+[^>]*>', '', text)
    # Normalize excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_rule_numbers(text):
    if not text:
        return []
    matches = RULE_REGEX.findall(text)
    valid_rules = []
    for m in matches:
        m_str = m.strip()
        parts = m_str.split('.')
        if len(parts) == 2 or len(parts) == 3:
            try:
                major = int(parts[0])
                if 1 <= major <= 55: # Up Front rule numbers range 1-55
                    valid_rules.append(m_str)
            except ValueError:
                pass
    return sorted(list(set(valid_rules)))

def extract_scenarios(text):
    if not text:
        return []
    matches = SCENARIO_REGEX.findall(text)
    return sorted(list(set(matches)))

def extract_concepts(text):
    text_lower = text.lower()
    detected = []
    for concept, keywords in CORE_CONCEPTS.items():
        if any(kw in text_lower for kw in keywords):
            detected.append(concept)
    return detected

def classify_intent(title, question_text, rule_numbers, scenarios):
    combined = f"{title} {question_text}".lower()
    
    if scenarios or "scenario" in combined or "victory condition" in combined:
        return "scenario"
    elif "errata" in combined or "version" in combined or "update" in combined or "edition" in combined:
        return "errata_comparison"
    elif len(rule_numbers) > 0 and any(f"rule {r}" in combined or f"[{r}]" in combined or f"({r})" in combined for r in rule_numbers):
        return "direct_rule"
    elif any(phrase in combined for phrase in ["can i", "what happens", "is it legal", "if a player", "does this mean", "how do you resolve"]):
        return "situation"
    elif any(phrase in combined for phrase in ["how does", "what is", "how do", "clarification on", "meaning of"]):
        return "concept"
    else:
        return "clarification"

def fetch_thread_details(thread_header):
    tid = thread_header['threadid']
    url = f"https://api.geekdo.com/api/articles?threadid={tid}"
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                articles = data.get('articles', [])
                if not articles:
                    return None
                    
                # OP Question
                op = articles[0]
                user_obj = thread_header.get('user')
                op_author = user_obj.get('username') if isinstance(user_obj, dict) else str(op.get('author', 'Unknown'))
                op_date = op.get('postdate', '')
                op_body = clean_text(op.get('body', ''))
                
                # Replies
                replies = []
                for a in articles[1:]:
                    auth = a.get('author')
                    r_author = str(auth) if auth is not None else "Unknown"
                    r_date = a.get('postdate', '')
                    r_body = clean_text(a.get('body', ''))
                    if r_body:
                        replies.append({
                            "author_id": r_author,
                            "date": r_date,
                            "body": r_body,
                            "rules_cited": extract_rule_numbers(r_body)
                        })
                        
                # Extracted metadata
                title = thread_header.get('subject', '').strip()
                all_text_for_rules = f"{title}\n{op_body}\n" + "\n".join(r['body'] for r in replies)
                rule_citations = extract_rule_numbers(all_text_for_rules)
                scenarios = extract_scenarios(all_text_for_rules)
                concepts = extract_concepts(all_text_for_rules)
                intent = classify_intent(title, op_body, rule_citations, scenarios)
                
                # Best consensus answer selection:
                # 1. Prefer reply citing exact rules
                # 2. Or longest informative reply
                consensus_answer = ""
                if replies:
                    replies_with_rules = [r for r in replies if len(r['rules_cited']) > 0]
                    if replies_with_rules:
                        # Sort by rule count then length
                        replies_with_rules.sort(key=lambda r: (len(r['rules_cited']), len(r['body'])), reverse=True)
                        consensus_answer = replies_with_rules[0]['body']
                    else:
                        # Pick longest reply
                        replies.sort(key=lambda r: len(r['body']), reverse=True)
                        consensus_answer = replies[0]['body']

                item = {
                    "id": f"bgg_uf_{tid}",
                    "thread_id": tid,
                    "title": title,
                    "question": op_body,
                    "author": op_author,
                    "post_date": op_date,
                    "url": f"https://boardgamegeek.com/thread/{tid}",
                    "intent": intent,
                    "rule_citations": rule_citations,
                    "scenarios_mentioned": scenarios,
                    "concepts": concepts,
                    "num_replies": len(replies),
                    "consensus_answer": consensus_answer,
                    "replies": replies
                }
                return item
        except Exception as e:
            if attempt == 2:
                print(f"Error fetching thread {tid}: {e}")
            time.sleep(0.5)
            
    return None

def main():
    headers_file = "data/eval/bgg_upfront_threads_headers.json"
    if not os.path.exists(headers_file):
        raise FileNotFoundError(f"Thread headers file not found at {headers_file}")
        
    with open(headers_file, "r", encoding="utf-8") as f:
        threads = json.load(f)
        
    print(f"Loaded {len(threads)} thread headers. Fetching articles concurrently...")
    
    eval_set = []
    completed = 0
    total = len(threads)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_thread_details, t): t for t in threads}
        for future in as_completed(futures):
            res = future.result()
            if res and res.get('question'):
                eval_set.append(res)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"Progress: {completed}/{total} threads fetched ({len(eval_set)} valid QA entries)")
                
    # Sort deterministically by thread_id descending
    eval_set.sort(key=lambda x: int(x['thread_id']), reverse=True)
    
    print(f"\nSuccessfully compiled {len(eval_set)} rules evaluation entries!")
    
    # Save Full Dataset
    os.makedirs("data/eval", exist_ok=True)
    full_path = "data/eval/upfront_bgg_eval_set_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)
    print(f"Saved full eval dataset to {full_path}")
    
    # Save Streamlined Benchmark QA Set (Optimized for Automated RAG Evaluation)
    benchmark_set = []
    for item in eval_set:
        # Create clear query combining title and question
        q_combined = item['title']
        if item['question'] and item['question'].lower() not in item['title'].lower():
            q_combined = f"{item['title']}: {item['question']}"
            
        benchmark_set.append({
            "id": item["id"],
            "query": q_combined,
            "title": item["title"],
            "raw_question": item["question"],
            "intent": item["intent"],
            "expected_rule_citations": item["rule_citations"],
            "scenarios": item["scenarios_mentioned"],
            "concepts": item["concepts"],
            "ground_truth_answer": item["consensus_answer"],
            "num_replies": item["num_replies"],
            "source_url": item["url"]
        })
        
    bench_path = "data/eval/upfront_bgg_eval_benchmark.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_set, f, indent=2)
    print(f"Saved benchmark QA dataset to {bench_path}")

    # Generate Statistical Breakdown Summary Markdown
    total_q = len(benchmark_set)
    intent_counts = {}
    rule_counts = {}
    concept_counts = {}
    with_citations = 0
    with_answers = 0
    
    for item in benchmark_set:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        if item['expected_rule_citations']:
            with_citations += 1
            for r in item['expected_rule_citations']:
                rule_counts[r] = rule_counts.get(r, 0) + 1
                
        if item['ground_truth_answer']:
            with_answers += 1
            
        for c in item['concepts']:
            concept_counts[c] = concept_counts.get(c, 0) + 1
            
    top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    
    summary_md = f"""# Up Front BGG Rules Forum Evaluation Dataset Summary

This evaluation benchmark was curated from all threads in the official **BoardGameGeek Up Front Rules Forum** (Forum ID: 66, Game ID: 586).

## 📊 Dataset Metrics
- **Total Questions / Threads**: {total_q}
- **Questions with Expert Community Answers**: {with_answers} ({with_answers/total_q*100:.1f}%)
- **Questions with Explicit Rule Number Citations**: {with_citations} ({with_citations/total_q*100:.1f}%)
- **Unique Rule Numbers Cited Across Corpus**: {len(rule_counts)}

## 🏷️ Distribution by Query Intent
| Intent Category | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
"""
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        summary_md += f"| `{intent}` | {count} | {count/total_q*100:.1f}% | Rules query category |\n"
        
    summary_md += f"""
## 🎯 Top 30 Most Frequently Inquired Rules
| Rule Number | Inquiries / Citations | Primary Topic Area |
| :---: | :---: | :--- |
"""
    for r, count in top_rules:
        summary_md += f"| `{r}` | {count} | Technical Up Front Rule |\n"
        
    summary_md += f"""
## 🧠 Concept Distribution
| Concept Area | Questions Tagged |
| :--- | :---: |
"""
    for c, count in top_concepts:
        summary_md += f"| `{c}` | {count} |\n"
        
    summary_path = "data/eval/upfront_bgg_eval_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Saved dataset summary to {summary_path}")

if __name__ == "__main__":
    main()
