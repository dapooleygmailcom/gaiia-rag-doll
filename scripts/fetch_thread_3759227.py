import urllib.request
import json
import html

url = 'https://api.geekdo.com/api/articles?threadid=3759227'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json, text/plain, */*'
}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=12) as resp:
    data = json.loads(resp.read().decode('utf-8'))

articles = data.get('articles', [])
print(f"Total articles: {len(articles)}")
for i, a in enumerate(articles):
    print(f"=== Article {i+1} by {a.get('author')} ({a.get('postdate')}) ===")
    body = html.unescape(a.get('body', ''))
    print(body)
    print("\n" + "="*50 + "\n")
