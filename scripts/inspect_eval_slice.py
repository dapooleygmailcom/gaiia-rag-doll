import json

def inspect_slice():
    with open("data/eval/upfront_bgg_eval_benchmark.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    for i in range(60, 75):
        item = items[i]
        print(f"Test #{i+1} ID: {item['id']}")
        print(f"  Query: {item['query']}")
        print(f"  Expected Rules: {item.get('expected_rule_citations')}")
        print("---")

if __name__ == "__main__":
    inspect_slice()
