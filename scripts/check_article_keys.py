import urllib.request, json
url = 'https://api.geekdo.com/api/articles?threadid=3679123'
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    for i, a in enumerate(data.get('articles', [])):
        print(f"Article {i}: keys={list(a.keys())}")
        print(f"  id={a.get('id')} postdate={a.get('postdate')} numrecommend={a.get('numrecommend')} user={a.get('user')}")
