import json

loc_map = {r['id']: r for r in json.load(open('data/eval/upfront_bgg_eval_checkpoint.json', encoding='utf-8'))['results']}
cld_map = {r['id']: r for r in json.load(open('data/eval/upfront_cloud_eval_1_100.json', encoding='utf-8'))['results'] if r.get('overall_index', 0) <= 60}

miss_ids = [
    ('bgg_uf_3645331', 6),
    ('bgg_uf_3589681', 14),
    ('bgg_uf_3377439', 24),
    ('bgg_uf_3371642', 25),
    ('bgg_uf_3199417', 37),
    ('bgg_uf_3164229', 42),
    ('bgg_uf_2984341', 54),
]

for mid, idx in miss_ids:
    l = loc_map.get(mid, {})
    c = cld_map.get(mid, {})
    print("=" * 75)
    print(f"#{idx} [{mid}]: {l.get('title')}")
    print(f"Query: {l.get('query')[:120]}...")
    print(f"Expected Rules: {l.get('expected_rules')}")
    print(f"Local Hits: {l.get('hits')} | Local Recall: {l.get('rule_recall')}")
    all_l_ret = l.get('retrieved_rules', [])
    hit_ranks = [all_l_ret.index(h) + 1 for h in l.get('hits', []) if h in all_l_ret]
    print(f"Local Hit Rank Position in retrieved_rules: {hit_ranks} out of {len(all_l_ret)}")
    print(f"Local Retrieved Rules ({len(all_l_ret)} total): {all_l_ret[:15]}")
    print(f"Cloud Hits: {c.get('hits')} | Cloud Recall: {c.get('rule_recall')}")
    print(f"Cloud Retrieved Rules ({len(c.get('retrieved_rules', []))} total): {c.get('retrieved_rules', [])}")
    print(f"Cloud Verdict: {c.get('verdict')}")
    print(f"Local Latency: {l.get('latency_seconds')}s | Cloud Latency: {c.get('latency_seconds')}s")
    
    ld = l.get('debug', {}) or {}
    cd = c.get('debug', {}) or {}
    print("\n--- Local Debug ---")
    for k in ['query_type', 'rule_numbers', 'sub_queries', 'num_retrieved', 'num_parent_expansions', 'num_cooccurrence_expansions', 'num_adjacent_expansions', 'num_cross_refs', 'hyde_clause']:
        if k in ld:
            print(f"  {k}: {ld[k]}")
print("\n" + "="*75)
print("SUMMARY OF LOCAL RANKS FOR THE 7 DISCREPANCY CASES:")
print("="*75)
for mid, idx in miss_ids:
    l = loc_map.get(mid, {})
    c = cld_map.get(mid, {})
    all_l_ret = l.get('retrieved_rules', [])
    hits = l.get('hits', [])
    hit_ranks = [all_l_ret.index(h) + 1 for h in hits if h in all_l_ret]
    c_ret = c.get('retrieved_rules', [])
    print(f"#{idx:<2} [{mid}] Exp: {str(l.get('expected_rules')):<15} | Local Hits: {str(hits):<12} | Local Rank: {str(hit_ranks):<8} of {len(all_l_ret):<3} | Cloud Citations Returned: {len(c_ret)}")

