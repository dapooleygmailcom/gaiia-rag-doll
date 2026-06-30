import fitz
import re

def extract_acronym_candidates(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + " "
    doc.close()
    
    # Common acronym definition patterns
    patterns = [
        # Full Name (ACR)
        r'([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})\s*\(([A-Z]{2,5}[a-z]?)\)',
        # ACR (Full Name)
        r'([A-Z]{2,5}[a-z]?)\s*\(([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})\)',
        # Full Name [ACR]
        r'([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})\s*\[([A-Z]{2,5}[a-z]?)\]',
        # ACR: Full Name
        r'\b([A-Z]{2,5}[a-z]?):\s*([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,4})'
    ]
    
    candidates = set()
    for pat_str in patterns:
        pat = re.compile(pat_str)
        for match in pat.finditer(text):
            # Try to figure out which group is the acronym (usually the shorter one in all caps)
            g1, g2 = match.groups()
            g1 = g1.strip()
            g2 = g2.strip()
            
            # Simple heuristic: Acronym is mostly uppercase and shorter
            if len(g1) <= 6 and g1.isupper() and len(g2) > len(g1):
                candidates.add(f"{g1}: {g2}")
            elif len(g2) <= 6 and g2.isupper() and len(g1) > len(g2):
                candidates.add(f"{g2}: {g1}")
                
    return sorted(list(candidates))

if __name__ == "__main__":
    cands = extract_acronym_candidates("data/asl/pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf")
    print(f"Found {len(cands)} candidates:")
    for c in cands[:20]:
        print("  -", c)
    print("...")
    for c in cands[-20:]:
        print("  -", c)
