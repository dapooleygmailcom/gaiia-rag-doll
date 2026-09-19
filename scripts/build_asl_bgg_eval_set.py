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

# ASL Chapters: A through W, Z, plus ASOP and IIFT
VALID_CHAPTERS = set("ABCDEFGHIJKLMNOPQRSTUVWZ")

# Match Chapter-Decimal rule patterns like A7.212, B13.3, C8.52, D2.11, E1.1, G1.42, J2.3, O11.4, ASOP 1.1A
ASL_RULE_DECIMAL_REGEX = re.compile(
    r'(?:^|[\s\[\(\{\:;,\/])([A-W]\d{1,2}\.\d{1,5}(?:[a-zA-Z])?)(?:[\]\)\}\:;,\.\s\/]|$)',
    re.MULTILINE
)
# Match explicit whole-chapter rule citations like "Rule A10", "section C5", "[A10]", "(B23)"
ASL_RULE_EXPLICIT_REGEX = re.compile(
    r'(?:(?:\b(?:rule|rules|section|chapter|sec\.|cfr|see)\s+)|[\[\(])([A-W]\d{1,2})(?:[\]\)\:\,\.\s]|$)',
    re.IGNORECASE | re.MULTILINE
)

SCENARIO_REGEX = re.compile(
    r'\b(?:[Ss]cenarios?|ASL)\s+([A-Z0-9\-]+)\b|\b(?:AP|DASL|BFP|SP|WO|FT|DB|RO|OA|J)\s*#?\s*(\d+[A-Za-z]?)\b',
    re.IGNORECASE
)

CORE_CONCEPTS = {
    "movement_bypass": [
        "bypass", "bypass movement", "mp", "movement factor", "mf", "dash", "advance",
        "cx", "minimum move", "motion", "reverse", "stalking", "creep", "non-assault move", "assault move"
    ],
    "fire_attacks": [
        "first fire", "subsequent first fire", "sff", "final protective fire", "fpf",
        "prep fire", "defensive fire", "advancing fire", "fire group", "fg", "residual fp",
        "residual firepower", "rfp", "ift", "iift", "cowering", "cower", "firepower", "modified fp", "iiFT"
    ],
    "morale_pin_rally": [
        "morale check", "mc", "nmc", "1mc", "2mc", "3mc", "4mc", "pin", "pinned",
        "ptc", "broken", "rout", "rout phase", "rally", "self rally", "desperation morale",
        "dm", "elr", "heat of battle", "hob", "berserk", "surrender", "interdiction", "failure to rout"
    ],
    "terrain_los": [
        "los", "line of sight", "hindrance", "blind hex", "orchard", "woods", "building",
        "rubble", "grain", "brush", "wall", "hedge", "bocage", "crest", "elevation", "slope",
        "water obstacle", "smoke", "ffnam", "ffmo", "tem", "foxhole", "trench", "pillbox",
        "wire", "entrenchment", "road", "bridge", "gully", "stream", "hexside"
    ],
    "close_combat_melee": [
        "close combat", "cc", "melee", "ambush", "hand to hand", "hand-to-hand", "hth",
        "street fighting", "capture", "withdrawal from melee", "prisoner", "interrogation"
    ],
    "concealment_snipers": [
        "concealment", "concealed", "gain concealment", "loss of concealment", "dummy",
        "cloaking", "sniper", "san", "sniper check", "hip", "hidden initial placement"
    ],
    "ordnance_guns": [
        "to hit", "th table", "to hit table", "critical hit", "ch", "acquisition", "bore sighting",
        "gun malfunction", "breakdown", "repair", "rate of fire", "rof", "intensive fire",
        "mortar", "at gun", "ap", "he", "heat", "wp", "smoke", "special ammo", "app", "apcr",
        "apds", "canister", "cal", "caliber", "gun crew", "m-kill", "k-kill", "critical hit"
    ],
    "vehicles_afv": [
        "afv", "tank", "panzer", "sherman", "t-34", "tiger", "halftrack", "armored car",
        "motion", "stopped", "start mp", "stop mp", "vca", "tca", "hull", "turret", "armor",
        "armour", "shock", "unconfirmed kill", "burning wreck", "blaze", "bog", "bog check",
        "mirable", "overrun", "ovr", "passengers", "riders", "ce", "bu", "buttoned up",
        "open topped", "sd", "sn", "smoke discharger", "hull down", "hd"
    ],
    "leaders_heroes": [
        "leader", "leader drm", "leader direction", "leader loss", "wound", "wounded leader",
        "hero", "heroic", "hero creation", "commissar", "nco", "officer", "sniper target", "battle hardened"
    ],
    "scenarios_setup": [
        "setup", "reinforcements", "entry", "san", "elr", "turn limit", "victory condition",
        "victory conditions", "vc", "ssr", "special scenario rule", "special scenario rules",
        "balance", "night", "fog", "rain", "snow", "wind", "weather", "eto", "pto", "desert"
    ]
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
    
    valid_rules = set()
    
    # 1. Chapter decimal rules (A7.212, B13.3, C8.52, D2.11, etc.)
    for m in ASL_RULE_DECIMAL_REGEX.findall(text):
        m_clean = m.strip().rstrip('.')
        if len(m_clean) >= 3 and m_clean[0].upper() in VALID_CHAPTERS:
            valid_rules.add(m_clean)
            
    # 2. Explicit rules (Rule A10, [A10], etc.)
    for m in ASL_RULE_EXPLICIT_REGEX.findall(text):
        m_clean = m.strip()
        if len(m_clean) >= 2 and m_clean[0].upper() in VALID_CHAPTERS:
            valid_rules.add(m_clean)

    # Sort naturally
    return sorted(list(valid_rules))

def extract_scenarios(text):
    if not text:
        return []
    matches = SCENARIO_REGEX.findall(text)
    cleaned = set()
    for m in matches:
        if isinstance(m, tuple):
            val = next((item for item in m if item), "")
        else:
            val = m
        val_clean = val.strip().rstrip(".,;:")
        if 1 <= len(val_clean) <= 10:
            cleaned.add(val_clean)
    return sorted(list(cleaned))

# Precompile concept regexes with word boundaries
CONCEPT_PATTERNS = {
    concept: [
        re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        for kw in keywords
    ]
    for concept, keywords in CORE_CONCEPTS.items()
}

def extract_concepts(text):
    if not text:
        return []
    detected = []
    for concept, patterns in CONCEPT_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            detected.append(concept)
    return detected

def classify_intent(title, question_text, rule_numbers, scenarios):
    combined = f"{title} {question_text}".lower()
    
    if scenarios or "scenario" in combined or "ssr" in combined or "victory condition" in combined:
        return "scenario"
    elif "errata" in combined or "version" in combined or "update" in combined or "edition" in combined or "2nd edition" in combined:
        return "errata_comparison"
    elif len(rule_numbers) > 0 and any(f"rule {r.lower()}" in combined or f"[{r.lower()}]" in combined or f"({r.lower()})" in combined for r in rule_numbers):
        return "direct_rule"
    elif any(phrase in combined for phrase in ["can i", "what happens", "is it legal", "if a player", "does this mean", "how do you resolve", "is an afv", "can a unit"]):
        return "situation"
    elif any(phrase in combined for phrase in ["how does", "what is", "how do", "clarification on", "meaning of", "difference between"]):
        return "concept"
    else:
        return "clarification"

def fetch_thread_details(thread_header):
    tid = thread_header['threadid']
    url = f"https://api.geekdo.com/api/articles?threadid={tid}"
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
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
                    "id": f"bgg_asl_{tid}",
                    "thread_id": str(tid),
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
    headers_file = "data/eval/bgg_asl_threads_headers.json"
    if not os.path.exists(headers_file):
        raise FileNotFoundError(f"Thread headers file not found at {headers_file}")
        
    with open(headers_file, "r", encoding="utf-8") as f:
        threads = json.load(f)
        
    print(f"Loaded {len(threads)} ASL thread headers. Fetching articles concurrently...")
    
    eval_set = []
    completed = 0
    total = len(threads)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_thread_details, t): t for t in threads}
        for future in as_completed(futures):
            res = future.result()
            if res and res.get('question'):
                eval_set.append(res)
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} threads fetched ({len(eval_set)} valid QA entries)")
                
    # Sort deterministically by thread_id descending
    eval_set.sort(key=lambda x: int(x['thread_id']), reverse=True)
    
    print(f"\nSuccessfully compiled {len(eval_set)} ASL rules evaluation entries!")
    
    # Save Full Dataset
    os.makedirs("data/eval", exist_ok=True)
    full_path = "data/eval/asl_bgg_eval_set_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)
    print(f"Saved full ASL eval dataset to {full_path}")
    
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
        
    bench_path = "data/eval/asl_bgg_eval_benchmark.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_set, f, indent=2)
    print(f"Saved ASL benchmark QA dataset to {bench_path}")

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
            
    top_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:35]
    top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    
    summary_md = f"""# Advanced Squad Leader (ASL) BGG Rules Forum Evaluation Dataset Summary

This evaluation benchmark was curated from all threads in the official **BoardGameGeek Advanced Squad Leader Rules Forum** (Forum ID: 66, Game ID: 243).

## 📊 Dataset Metrics
- **Total Questions / Threads**: {total_q}
- **Questions with Expert Community Answers**: {with_answers} ({with_answers/total_q*100:.1f}%)
- **Questions with Explicit Rule Number Citations**: {with_citations} ({with_citations/total_q*100:.1f}%)
- **Unique ASL Rule Numbers Cited Across Corpus**: {len(rule_counts)}

## 🏷️ Distribution by Query Intent
| Intent Category | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
"""
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        summary_md += f"| `{intent}` | {count} | {count/total_q*100:.1f}% | Rules query category |\n"
        
    summary_md += f"""
## 🎯 Top 35 Most Frequently Inquired ASL Rules
| Rule Number | Inquiries / Citations | Primary Topic Area |
| :---: | :---: | :--- |
"""
    for r, count in top_rules:
        summary_md += f"| `{r}` | {count} | Technical ASL Rule |\n"
        
    summary_md += f"""
## 🧠 Concept Distribution
| Concept Area | Questions Tagged |
| :--- | :---: |
"""
    for c, count in top_concepts:
        summary_md += f"| `{c}` | {count} |\n"
        
    summary_path = "data/eval/asl_bgg_eval_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Saved dataset summary to {summary_path}")

if __name__ == "__main__":
    main()
