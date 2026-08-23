import urllib.request
import urllib.parse
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://boardgamegeek.com/thread/3720028'
}

thread_endpoints = [
    'https://api.geekdo.com/api/articles?threadid=3720028',
    'https://api.geekdo.com/api/forums/articles?threadid=3720028',
    'https://api.geekdo.com/api/threads/3720028',
    'https://api.geekdo.com/api/forumthreads/3720028',
    'https://api.geekdo.com/api/forums/threads/3720028',
    'https://api.geekdo.com/api/articles/3720028',
    'https://api.geekdo.com/api/articles?objectid=3720028&objecttype=thread',
    'https://api.geekdo.com/api/geekitems?objectid=3720028&objecttype=thread',
    'https://api.geekdo.com/api/thread/3720028'
]

for ep in thread_endpoints:
    try:
        req = urllib.request.Request(ep, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode('utf-8')
            print(f"SUCCESS! {ep} -> Status: {resp.status}, Len: {len(data)}")
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                print(f"Keys: {list(parsed.keys())}")
                if 'articles' in parsed:
                    print(f"Articles in response: {len(parsed['articles'])}")
                    print("Sample article body:")
                    sample_art = parsed['articles'][0]
                    print(f"Author: {sample_art.get('username')}, Date: {sample_art.get('postdate')}")
                    print(f"Body: {sample_art.get('body', '')[:300]}")
            elif isinstance(parsed, list):
                print(f"List response len: {len(parsed)}")
                if len(parsed) > 0:
                    print("Sample item:", json.dumps(parsed[0], indent=2)[:300])
            print("="*60)
    except Exception as e:
        print(f"Failed {ep}: {e}")
