import os
import json
import fitz

data_dir = "data/asl"
files = os.listdir(data_dir)

with open("data/asl_profile.json", encoding="utf-8") as f:
    profile = json.load(f)

docs = profile.get("documents", {})

for fname in sorted(files):
    if not fname.endswith(".pdf") and not fname.endswith(".txt"):
        continue
    fpath = os.path.join(data_dir, fname)
    size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
    
    total_pages = 1
    if fname.endswith(".pdf"):
        try:
            doc = fitz.open(fpath)
            total_pages = len(doc)
            doc.close()
        except Exception:
            pass

    if "2nd-edition-core-rules" in fname:
        docs[fname] = {
            "doc_type": "core_rules",
            "priority": 1,
            "description": "ASL 2nd Edition Core Rulebook - primary authority",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }
    elif "1st-edition" in fname:
        docs[fname] = {
            "doc_type": "core_rules_v1",
            "priority": 9,
            "description": "ASL 1st Edition Rulebook (superseded by 2nd ed)",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": 0  # skip to avoid duplicate 1st ed noise
        }
    elif "AOO_2006" in fname:
        docs[fname] = {
            "doc_type": "core_rules",
            "priority": 1,
            "description": "Armies of Oblivion Core Chapter Pages (2006)",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }
    elif "Version_Tracker" in fname:
        docs[fname] = {
            "doc_type": "version_tracker",
            "priority": 2,
            "description": "ASL Rulebook Version Tracker",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": 50
        }
    elif "Scenario_Errata" in fname:
        docs[fname] = {
            "doc_type": "scenario_errata",
            "priority": 4,
            "description": "Scenario-specific published errata (Nov 2025)",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }
    elif "Scenario_Balance" in fname:
        docs[fname] = {
            "doc_type": "scenario_balance",
            "priority": 5,
            "description": "Scenario balance adjustments (Nov 2025)",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }
    elif "Errata" in fname or "errata" in fname or fname.endswith("vB.pdf"):
        docs[fname] = {
            "doc_type": "errata",
            "priority": 3,
            "description": "Official Errata Document / Replacement Pages",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }
    elif "QA" in fname or "Q&A" in fname:
        docs[fname] = {
            "doc_type": "qa",
            "priority": 6,
            "description": "ASL Q&A Clarifications",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }
    elif "journal" in fname.lower():
        docs[fname] = {
            "doc_type": "journal",
            "priority": 6,
            "description": "ASL Journal Magazine",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": 50
        }
    elif fname.startswith("Scenario") or fname.startswith("ap") or fname.startswith("pp") or "scenarios" in fname.lower():
        docs[fname] = {
            "doc_type": "scenarios",
            "priority": 8,
            "description": "Scenario Cards / Packs",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }
    elif fname not in docs:
        docs[fname] = {
            "doc_type": "unknown",
            "priority": 9,
            "description": "Unknown document",
            "size_mb": size_mb,
            "total_pages": total_pages,
            "max_pages": None
        }

profile["documents"] = docs
profile["cooccurrence_graph_file"] = "data/asl_cooccurrence_graph.json"
profile["section_tree_file"] = "data/asl_section_tree.json"

with open("data/asl_profile.json", "w", encoding="utf-8") as f:
    json.dump(profile, f, indent=2)

print(f"Updated asl_profile.json with {len(docs)} documents.")
active_count = sum(1 for d in docs.values() if d.get("max_pages") != 0)
print(f"Active documents: {active_count}")
for k, v in sorted(docs.items(), key=lambda x: x[1]["priority"])[:15]:
    print(f"  [P{v['priority']}] {v['doc_type']:16s} (pages: {v.get('max_pages') or 'all'}) {k}")
