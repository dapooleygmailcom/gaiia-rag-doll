import fitz
import os

DATA_DIR = "data/upfront"

# Check which PDFs have extractable text vs are scanned images
for fname in os.listdir(DATA_DIR):
    if not fname.endswith('.pdf'):
        continue
    fpath = os.path.join(DATA_DIR, fname)
    doc = fitz.open(fpath)
    
    total_chars = 0
    has_images = False
    for i in range(min(5, len(doc))):
        text = doc[i].get_text()
        total_chars += len(text.strip())
        imgs = doc[i].get_images()
        if imgs:
            has_images = True
    
    status = "TEXT" if total_chars > 100 else ("SCANNED/IMAGE" if has_images else "EMPTY")
    print(f"{fname:<55} pages={len(doc):>3}  chars(first5)={total_chars:>6}  status={status}")
    doc.close()

# Now read the core rules doc (Up_Front.pdf) more deeply
print("\n\n=== DEEP READ: Up_Front.pdf (core rules) ===")
doc = fitz.open(os.path.join(DATA_DIR, "Up_Front.pdf"))
for i in range(min(8, len(doc))):
    text = doc[i].get_text()
    if len(text.strip()) > 50:
        print(f"\n--- Page {i+1} ---")
        print(text[:2000])
doc.close()

# Read errata  
print("\n\n=== DEEP READ: Up_Front_Errata_Pages.pdf ===")
doc = fitz.open(os.path.join(DATA_DIR, "Up_Front_Errata_Pages.pdf"))
for i in range(min(5, len(doc))):
    text = doc[i].get_text()
    if len(text.strip()) > 10:
        print(f"\n--- Page {i+1} ---")
        print(text[:3000])
doc.close()

# Read the updated rulebook deeper
print("\n\n=== DEEP READ: UF RuleBook updated.pdf ===")
doc = fitz.open(os.path.join(DATA_DIR, "UF RuleBook updated.pdf"))
for i in range(min(10, len(doc))):
    text = doc[i].get_text()
    if len(text.strip()) > 50:
        print(f"\n--- Page {i+1} ---")
        print(text[:2000])
doc.close()
