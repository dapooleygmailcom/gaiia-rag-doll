import urllib.request, json
url = 'https://api.geekdo.com/api/articles?threadid=3720028'
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))
    art0 = data['articles'][0]
    print(json.dumps(art0, indent=2))
