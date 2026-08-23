import urllib.request
import urllib.error
import json
import xml.etree.ElementTree as ET

urls = [
    ("XMLAPI2 Forumlist", "https://boardgamegeek.com/xmlapi2/forumlist?id=586&type=thing"),
    ("XMLAPI1 Boardgame", "https://boardgamegeek.com/xmlapi/boardgame/586"),
    ("Geekdo Forum API", "https://api.geekdo.com/api/forums?objectid=586&objecttype=thing"),
    ("Geekdo Direct Forum 66", "https://api.geekdo.com/api/threads?forumid=66&objectid=586&objecttype=thing"),
    ("BGG HTML Forum", "https://boardgamegeek.com/forum/66/up-front/rules")
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

for name, url in urls:
    print(f"\n--- Testing: {name} ({url}) ---")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            print(f"Status: {resp.status}, Length: {len(data)}")
            preview = data[:300].decode('utf-8', errors='replace')
            print(f"Preview: {preview}")
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} - {e.reason}")
    except Exception as e:
        print(f"Error: {e}")

