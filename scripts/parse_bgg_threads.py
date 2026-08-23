import os
import re
import urllib.request
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

url = "https://boardgamegeek.com/boardgame/586/up-front/forums/66"
print(f"Fetching: {url}")
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        print(f"Downloaded {len(html)} bytes")
        
        # Save HTML
        with open("data/bgg_uf_rules_page1.html", "w", encoding="utf-8") as f:
            f.write(html)
            
        title = re.findall(r'<title>(.*?)</title>', html, re.I)
        print("Page Title:", title)
        
        # Find threads
        threads = re.findall(r'href=["\'](/thread/\d+/[^"\']+)["\']', html)
        print(f"Found {len(threads)} thread links")
        unique_threads = list(dict.fromkeys(threads))
        print(f"Unique thread links: {len(unique_threads)}")
        for t in unique_threads[:15]:
            print("  ", t)
            
        # Find pagination links
        pagination = re.findall(r'href=["\']([^"\']*page[^"\']*)["\']', html)
        print("\nPagination matches:", list(set(pagination))[:10])

except Exception as e:
    print(f"Error fetching: {e}")
