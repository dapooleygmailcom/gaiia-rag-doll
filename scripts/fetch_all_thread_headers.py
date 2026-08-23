import urllib.request
import urllib.parse
import json
import time
import re
import os

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://boardgamegeek.com/boardgame/586/up-front/forums/66'
}

def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to fetch {url}: {e}")
                return None
            time.sleep(1.0)
    return None

def fetch_all_thread_headers():
    print("Fetching all thread headers from BGG Up Front Rules forum (forumid=66)...")
    all_threads = []
    seen_ids = set()
    pageid = 1
    
    while True:
        url = f"https://api.geekdo.com/api/forums/threads?objectid=586&objecttype=thing&forumid=66&pageid={pageid}"
        data = fetch_json(url)
        if not data or 'threads' not in data:
            print(f"Page {pageid} had no valid data. Ending scan.")
            break
            
        threads = data['threads']
        if not threads:
            print(f"Page {pageid} returned empty list. Scan complete.")
            break
            
        new_count = 0
        for t in threads:
            tid = t.get('threadid')
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                all_threads.append(t)
                new_count += 1
                
        print(f"Page {pageid}: fetched {len(threads)} threads ({new_count} new, total unique: {len(all_threads)})")
        if new_count == 0:
            print("No new threads found on page. Stopping.")
            break
            
        pageid += 1
        time.sleep(0.2)
        
    return all_threads

if __name__ == "__main__":
    threads = fetch_all_thread_headers()
    print(f"\nTotal Unique Threads Collected: {len(threads)}")
    os.makedirs("data/eval", exist_ok=True)
    out_path = "data/eval/bgg_upfront_threads_headers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(threads, f, indent=2)
    print(f"Saved thread headers to {out_path}")
