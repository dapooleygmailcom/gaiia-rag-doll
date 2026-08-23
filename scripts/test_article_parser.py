import urllib.request
import json
import re
import html

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
}

def clean_bbcode_and_html(raw_text):
    if not raw_text:
        return ""
    text = raw_text
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove quotes blocks or preserve them nicely
    text = re.sub(r'\[quote(?:=[^\]]*)?\](.*?)\[/quote\]', r'Quote: "\1"', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove BBCode tags like [b], [/b], [url], [i], etc.
    text = re.sub(r'\[/?[a-zA-Z0-9_-]+(?:=[^\]]*)?\]', '', text)
    # Remove HTML tags like <br>, <p>, etc.
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?[a-zA-Z0-9_-]+[^>]*>', '', text)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

with open("data/eval/bgg_upfront_threads_headers.json", "r", encoding="utf-8") as f:
    threads = json.load(f)

print(f"Loaded {len(threads)} thread headers. Testing first 3 threads...")

for t in threads[:3]:
    tid = t['threadid']
    url = f"https://api.geekdo.com/api/articles?threadid={tid}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        articles = data.get('articles', [])
        print(f"\n================ Thread {tid}: {t.get('subject')} ================")
        print(f"Total Articles: {len(articles)}")
        if articles:
            op = articles[0]
            print(f"OP ({op.get('username', 'Unknown')}):")
            print(clean_bbcode_and_html(op.get('body', ''))[:300])
            print("---")
            if len(articles) > 1:
                first_reply = articles[1]
                print(f"First Reply ({first_reply.get('username', 'Unknown')}):")
                print(clean_bbcode_and_html(first_reply.get('body', ''))[:300])
