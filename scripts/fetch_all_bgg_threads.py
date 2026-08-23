import urllib.request
import urllib.parse
import json
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://boardgamegeek.com/boardgame/586/up-front/forums/66'
}

all_threads = []
page = 1
max_pages = 50

print("Scanning Up Front Rules forum (forumid=66, objectid=586)...")
while page <= max_pages:
    url = f"https://api.geekdo.com/api/forums/threads?objectid=586&objecttype=thing&forumid=66&page={page}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            threads = data.get('threads', [])
            if not threads:
                print(f"Page {page} returned 0 threads. Reached end.")
                break
            all_threads.extend(threads)
            print(f"Page {page}: fetched {len(threads)} threads (Total accumulated: {len(all_threads)})")
            page += 1
            time.sleep(0.3) # Friendly rate limit
    except Exception as e:
        print(f"Error on page {page}: {e}")
        break

print(f"\nCompleted thread list scan! Total threads found: {len(all_threads)}")

# Deduplicate by threadid
unique_threads = {t['threadid']: t for t in all_threads}
print(f"Unique thread IDs: {len(unique_threads)}")

# Save thread catalog
with open("data/upfront_bgg_threads_catalog.json", "w", encoding="utf-8") as f:
    json.dump(list(unique_threads.values()), f, indent=2)
print("Saved thread catalog to data/upfront_bgg_threads_catalog.json")
