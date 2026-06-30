"""Patch the ASL profile with corrected classifications."""
import json

with open("data/asl_profile.json") as f:
    p = json.load(f)

fixes = {
    "ASL_Scenario_Errata_Nov_2025.pdf": ("scenario_errata", 4, "Scenario-specific published errata (Nov 2025)"),
    "ASL_Scenario_Balance_Nov_2025.pdf": ("scenario_balance", 5, "Scenario balance adjustments (Nov 2025)"),
    "pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf": ("core_rules", 1, "ASL 2nd Edition Core Rulebook - primary authority"),
    "SR ASL_QA v22.pdf": ("qa", 6, "Q&A clarifications v22 (2005) - unofficial unless in official source"),
}
for fname, (dt, pr, desc) in fixes.items():
    if fname in p["documents"]:
        p["documents"][fname]["doc_type"] = dt
        p["documents"][fname]["priority"] = pr
        p["documents"][fname]["description"] = desc

with open("data/asl_profile.json", "w") as f:
    json.dump(p, f, indent=2)

print("Profile corrections applied")
print("Rule schema:", p["rule_schema"])
print("Docs active:", sum(1 for d in p["documents"].values() if d.get("max_pages") != 0))
print("\nAuthority stack (sorted by priority):")
for fname, info in sorted(p["documents"].items(), key=lambda x: x[1]["priority"])[:10]:
    print(f"  P{info['priority']} {info['doc_type']:20s} {fname[:50]}")
