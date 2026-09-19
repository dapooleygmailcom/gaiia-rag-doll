import os
import sys
import boto3

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    session = boto3.Session(profile_name="aiia", region_name="ap-southeast-2")
    lam = session.client("lambda")
    paginator = lam.get_paginator("list_functions")
    ragdoll_funcs = []
    for page in paginator.paginate():
        for fn in page.get("Functions", []):
            name = fn.get("FunctionName", "")
            if "RagDoll" in name or "adjudicate" in name.lower():
                ragdoll_funcs.append({
                    "name": name,
                    "runtime": fn.get("Runtime"),
                    "timeout": fn.get("Timeout"),
                    "lastModified": fn.get("LastModified")
                })
    print(f"Found {len(ragdoll_funcs)} RagDoll functions:")
    for f in sorted(ragdoll_funcs, key=lambda x: x["name"]):
        print(f"  {f['name']} (timeout: {f['timeout']}s, lastMod: {f['lastModified']})")

if __name__ == "__main__":
    main()
