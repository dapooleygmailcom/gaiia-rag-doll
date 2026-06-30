import subprocess
import os

print("Starting Regression Test Suite...")

print("\n1. Deleting old Up Front log if it exists...")
log_file = "data/logs/rules_lawyer_test_results.json"
if os.path.exists(log_file):
    os.remove(log_file)
    print("Deleted.")

print("\n2. Ingesting Up Front via generic pipeline...")
subprocess.run(["venv\\Scripts\\python.exe", "ingest_rules.py", "--profile", "data/up_front_profile.json"], check=True)

print("\n3. Running test_rules_lawyer.py (Up Front tests)...")
subprocess.run(["venv\\Scripts\\python.exe", "test_rules_lawyer.py"], check=True)

print("\n4. Running test_asl_sample.py (ASL smoke tests)...")
subprocess.run(["venv\\Scripts\\python.exe", "test_asl_sample.py"], check=True)

print("\nRegression suite completed successfully!")
