import urllib.request
import urllib.parse
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://boardgamegeek.com/boardgame/586/up-front/forums/66'
}

params_to_test = [
    {"objectid": 586, "objecttype": "thing", "forumid": 66, "pageid": 2},
    {"objectid": 586, "objecttype": "thing", "forumid": 66, "page": 2},
    {"objectid": 586, "objecttype": "thing", "forumid": 66, "pagesize": 50},
    {"objectid": 586, "objecttype": "thing", "forumid": 66, "offset": 10},
    {"objectid": 586, "objecttype": "thing", "forumid": 66, "start": 10},
    {"objectid": 586, "objecttype": "thing", "forumid": 66, "p": 2},
    {"forumid": 66, "objectid": 586, "objecttype": "thing", "pageid": 2},
    {"objectid": 586, "objecttype": "thing", "forumid": 66, "limit": 50}
]

for p in params_to_test:
    query_str = urllib.parse.urlencode(p)
    url = f"https://api.geekdo.com/api/forums/threads?{query_str}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            threads = data.get('threads', [])
            first_id = threads[0]['threadid'] if threads else "None"
            first_subj = threads[0]['subject'] if threads else "None"
            print(f"Param: {query_str}")
            print(f"  Count: {len(threads)}, First Thread: [{first_id}] {first_subj}")
    except Exception as e:
        print(f"Param: {query_str} -> Error: {e}")
