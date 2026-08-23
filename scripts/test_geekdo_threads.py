import urllib.request
import urllib.parse
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://boardgamegeek.com/boardgame/586/up-front/forums/66'
}

endpoints = [
    'https://api.geekdo.com/api/forums/threads?objectid=586&objecttype=thing&forumid=66',
    'https://api.geekdo.com/api/forums/threads?objectid=586&objecttype=thing&forumid=66&page=1',
    'https://api.geekdo.com/api/forums/threads?forumid=66&pageid=1',
    'https://api.geekdo.com/api/forums/threads?forumuid=614',
    'https://api.geekdo.com/api/forums/threads?forumid=614',
    'https://api.geekdo.com/api/forum/614/threads',
    'https://api.geekdo.com/api/forum/66/threads',
    'https://api.geekdo.com/api/forums/threads?forumid=66'
]

for ep in endpoints:
    try:
        req = urllib.request.Request(ep, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode('utf-8')
            print(f"SUCCESS! {ep} -> Status: {resp.status}, Len: {len(data)}")
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                print(f"Keys: {list(parsed.keys())}")
                if 'threads' in parsed:
                    print(f"Threads in response: {len(parsed['threads'])}")
                    print("Sample thread:")
                    print(json.dumps(parsed['threads'][0], indent=2))
            elif isinstance(parsed, list):
                print(f"List response len: {len(parsed)}")
                if len(parsed) > 0:
                    print("First item:", json.dumps(parsed[0], indent=2))
            print("="*60)
    except Exception as e:
        print(f"Failed {ep}: {e}")
