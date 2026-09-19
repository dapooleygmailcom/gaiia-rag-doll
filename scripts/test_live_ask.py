import os
import sys
import json
import urllib.request
import urllib.error

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def test_api():
    url = "https://8ifjmds7mk.execute-api.ap-southeast-2.amazonaws.com/api/ask"
    payload = {
        "tenantSlug": "internal-test",
        "titleSlug": "up-front",
        "query": "Can an immobilized vehicle fire its weapons?",
        "captchaToken": "test-eval-suite-token"
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST"
    )

    try:
        print(f"Connecting to {url}...")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            ruling = data.get("ruling", {})
            verdict = data.get("verdict") or ruling.get("verdict")
            citations = data.get("citations") or ruling.get("citations", [])
            answer = data.get("answerMarkdown") or ruling.get("answer") or ruling.get("summary", "")
            exec_time = data.get("executionTimeMs")

            print(f"Status: {resp.status}")
            print(f"Verdict: {verdict}")
            print(f"Execution Time: {exec_time}ms")
            print(f"Citations count: {len(citations)}")
            for c in citations[:5]:
                print(f"  Citation: Rule {c.get('ruleNumber')} - {c.get('titleName', c.get('chapter', ''))}")
            print("Answer preview:")
            print(answer[:300])
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
