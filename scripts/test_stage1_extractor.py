import os
import re
import sys
import json
import fitz

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, PROJECT_ROOT)

from engine.ingestion.ingest_visual import extract_page_image, extract_ocr_from_pixmap, slugify

KNOWN_PHOTOGRAPHERS = [
    "Arny Freytag", "Stephen Wayda", "Richard Fegley", "Mizuno", "Gen Mishino",
    "Jarmo Pohjaniemi", "Byron Newman", "Ric Moore", "Sean Bolger", "J.R. Mounger",
    "Wesley Martens", "Sasha Eisenman", "Josh Ryan", "Autumn Sonnichsen", "David Mecey",
    "Kim Mizuno", "Mario Casilli", "Ken Marcus", "Guerin Blask", "Jeff Dunas"
]

def parse_filename_metadata(pdf_path):
    raw_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    cleaned = raw_stem
    cleaned = re.sub(r'_(?:Pdf_)?Xxx_(?:Adult_)?Magazine\w*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'_Xxxmagazineporn(?:\s*\(\d+\))?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'_Adultxxxmagazine\w*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\d+playboy', 'Playboy', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(\d+\)$', '', cleaned)
    cleaned = cleaned.strip('_').strip()
    
    year = None
    yr_matches = re.findall(r'(?:19\d\d|20\d\d)', cleaned)
    if yr_matches:
        year = int(yr_matches[-1])
    else:
        broken = re.search(r'(19\d|20\d)[\D]+(\d)', raw_stem)
        if broken:
            year = int(broken.group(1) + broken.group(2))
            
    issue_date = None
    month_match = re.search(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*(?:[-_/]\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*)?)[\s_-]*(\d{4})?', cleaned, re.IGNORECASE)
    if month_match:
        issue_date = month_match.group(0).replace('_', ' ').strip()
    elif year:
        num_m = re.search(r'(\d{2})[-_](\d{2})[-_](\d{4})|(\d{4})[-_](\d{2})[-_](\d{2})|(\d{2})[-_](\d{4})', cleaned)
        if num_m:
            issue_date = num_m.group(0).replace('_', '-')
            
    if not issue_date and year:
        issue_date = str(year)

    title_norm = cleaned.replace('_', ' ')
    if 'germany' in title_norm.lower():
        publication = 'Playboy Germany'
    else:
        publication = 'Playboy Special Editions'

    mag_title = re.sub(r'[\d_-]+', ' ', cleaned).strip()
    mag_title = re.sub(r'\s+', ' ', mag_title)
    if not mag_title.lower().startswith('playboy'):
        mag_title = f"Playboy's {mag_title}"
    else:
        mag_title = mag_title.replace('Playboys', "Playboy's").replace('Playboy ', "Playboy's ")

    return {
        "raw_stem": raw_stem,
        "clean_stem": cleaned,
        "filename": os.path.basename(pdf_path),
        "publication": publication,
        "magazine_title": mag_title,
        "year": year,
        "issue_date": issue_date or str(year or "Unknown")
    }


def extract_bio_cards_from_text(ocr_text, page_num):
    """Parse bio cards, vital stats, measurements, and photographers from directory text."""
    cards = []
    # Normalize text
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    
    # Split by blocks if model names are all caps or numbered
    # Matches patterns like:
    # "TIFFANY RICHARDSON Despite the excitement... She's 27, 5'5", 100 pounds, 34-23-34. Mizuno, 1, 3, 4, 5"
    # "Sara Stokes Stats: 5'8" · 115 lbs. · 34D-24-34"
    full_text = " ".join(lines)
    
    # Regex pattern matching bio segments
    # Find all occurrences of measurements
    meas_matches = list(re.finditer(r'(\b\d{2}[A-G]*\s*-\s*\d{2}\s*-\s*\d{2}\b)', full_text))
    
    for idx, mm in enumerate(meas_matches):
        meas_str = mm.group(1).replace(' ', '')
        start_idx = meas_matches[idx - 1].end() if idx > 0 else max(0, mm.start() - 250)
        end_idx = meas_matches[idx + 1].start() if idx + 1 < len(meas_matches) else min(len(full_text), mm.end() + 150)
        
        block = full_text[start_idx:end_idx]
        
        # Extract height
        h_match = re.search(r'(\b\d\s*[\'’`]\s*\d{1,2}\s*[\"”`]?\b)', block)
        height = h_match.group(1).replace(' ', '').replace('’', "'").replace('`', "'") if h_match else None
        
        # Extract weight
        w_match = re.search(r'(\b\d{2,3})\s*(?:lbs?|pounds|Ibs|bs)\b', block, re.IGNORECASE)
        weight = int(w_match.group(1)) if w_match else None
        
        # Extract Photographer
        photog = None
        for kp in KNOWN_PHOTOGRAPHERS:
            if kp.lower() in block.lower():
                photog = kp
                break
        if not photog:
            p_match = re.search(r'(?:photo(?:graphy)?\s*(?:by)?|photos\s*by)\s*([A-Z][a-z]+\s+[A-Z][a-z]+)', block, re.IGNORECASE)
            if p_match:
                photog = p_match.group(1)
                
        # Extract Referenced Pages
        pages_ref = []
        pg_m = re.findall(r'(?:pages?|p\.?|,\s*)(\d{1,3}(?:\s*-\s*\d{1,3})?)', block[mm.start()-start_idx:])
        for pm in pg_m:
            if '-' in pm:
                ps, pe = pm.split('-')
                try:
                    pages_ref.extend(range(int(ps), int(pe) + 1))
                except Exception:
                    pass
            else:
                try:
                    pages_ref.append(int(pm))
                except Exception:
                    pass
        pages_ref = sorted(set(p for p in pages_ref if 1 <= p <= 200))
        
        # Extract candidate model name: uppercase words before the stats
        pre_meas = full_text[start_idx:mm.start()]
        # Look for capitalized full names
        name_cand = None
        names_found = re.findall(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})+)\b', pre_meas)
        if names_found:
            name_cand = names_found[-1].title()
        else:
            names_mixed = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', pre_meas)
            if names_mixed:
                # filter out non-names
                valid = [n for n in names_mixed if not any(w in n.upper() for w in ["VITAL STATS", "BARE FACTS", "SPECIAL EDITION", "CYBER CLUB", "PLAYMATE OF", "MODEL OF"])]
                if valid:
                    name_cand = valid[-1]
                    
        if name_cand:
            cards.append({
                "model_name": name_cand,
                "height": height,
                "weight_lbs": weight,
                "measurements": meas_str,
                "photographer": photog,
                "referenced_pages": pages_ref,
                "directory_page": page_num,
                "bio_snippet": block[:180]
            })
            
    return cards


def extract_issue_catalog(pdf_path):
    """
    Complete Pass 1 Extraction:
    Processes Cover, Candidate TOC (P1-P6), Candidate Bio (last 10 pages), and Pictorial Openers.
    Emits an authoritative Issue Catalog.
    """
    meta = parse_filename_metadata(pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 1. Cover
    p0 = doc[0]
    _, _, pix0 = extract_page_image(p0, dpi=100)
    cover_ocr = extract_ocr_from_pixmap(pix0).strip()
    
    # Extract Cover Girl
    cover_girl = None
    cg_match = re.search(r'(?:COVER\s*GIRL|FEATURING|ON\s*THE\s*COVER)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', cover_ocr, re.IGNORECASE)
    if cg_match:
        cover_girl = cg_match.group(1).title()
        
    # 2. TOC Pages (Scan P1 to P6)
    toc_models = []
    for pno in range(1, min(7, total_pages - 1)):
        p = doc[pno]
        _, _, pix = extract_page_image(p, dpi=100)
        ocr = extract_ocr_from_pixmap(pix).strip()
        upper = ocr.upper()
        if any(kw in upper for kw in ["CONTENTS", "INHALT", "FEATURING", "INDEX", "SPECIAL EDITIONS", "GIRLS OF"]) or re.search(r'\b\d{1,3}\s+[A-Z][a-z]+', ocr):
            for line in ocr.split('\n'):
                line = line.strip()
                # e.g. "Sara Stokes 4", "Heather Rene 10", "rinRIGGSp.66", "Melissa Lynn Miller 10"
                m1 = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(\d{1,3})\b', line)
                m2 = re.search(r'^(\d{1,3})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', line)
                if m1:
                    name, pg = m1.group(1).strip(), int(m1.group(2))
                    if not any(w in name.upper() for w in ["PLAYBOY", "SPECIAL", "PAGE", "EDITION", "DAS AUTO"]):
                        toc_models.append({"model_name": name, "page_start": pg, "toc_page": pno})
                elif m2:
                    pg, name = int(m2.group(1)), m2.group(2).strip()
                    if not any(w in name.upper() for w in ["PLAYBOY", "SPECIAL", "PAGE", "EDITION", "DAS AUTO"]):
                        toc_models.append({"model_name": name, "page_start": pg, "toc_page": pno})
                        
    # 3. Bio Directory Pages (Scan last 10 pages)
    bio_cards = []
    start_back = max(1, total_pages - 10)
    for pno in range(start_back, total_pages - 1):
        p = doc[pno]
        _, _, pix = extract_page_image(p, dpi=100)
        ocr = extract_ocr_from_pixmap(pix).strip()
        cards = extract_bio_cards_from_text(ocr, pno)
        if cards:
            bio_cards.extend(cards)
            
    # 4. Global Model Registry Fusion
    models_registry = {}
    
    # Add from Bio Cards
    for card in bio_cards:
        name = card["model_name"]
        slug = slugify(name)
        models_registry[slug] = {
            "model_name": name,
            "slug": slug,
            "height": card["height"],
            "weight_lbs": card["weight_lbs"],
            "measurements": card["measurements"],
            "photographer": card["photographer"],
            "pages": card["referenced_pages"],
            "start_page": card["referenced_pages"][0] if card["referenced_pages"] else None,
            "end_page": card["referenced_pages"][-1] if card["referenced_pages"] else None,
            "is_cover_girl": bool(cover_girl and slugify(cover_girl) == slug),
            "bio_summary": card["bio_snippet"]
        }
        
    # Add from TOC
    sorted_toc = sorted(toc_models, key=lambda x: x["page_start"])
    for i, tm in enumerate(sorted_toc):
        name = tm["model_name"]
        slug = slugify(name)
        start_p = tm["page_start"]
        end_p = sorted_toc[i + 1]["page_start"] - 1 if i + 1 < len(sorted_toc) else min(start_p + 7, total_pages - 2)
        spread_pages = list(range(start_p, max(start_p, end_p) + 1))
        
        if slug not in models_registry:
            models_registry[slug] = {
                "model_name": name,
                "slug": slug,
                "height": None,
                "weight_lbs": None,
                "measurements": None,
                "photographer": None,
                "pages": spread_pages,
                "start_page": start_p,
                "end_page": end_p,
                "is_cover_girl": bool(cover_girl and slugify(cover_girl) == slug),
                "bio_summary": None
            }
        else:
            if not models_registry[slug]["pages"]:
                models_registry[slug]["pages"] = spread_pages
                models_registry[slug]["start_page"] = start_p
                models_registry[slug]["end_page"] = end_p

    # Add Cover Girl if not in registry
    if cover_girl:
        c_slug = slugify(cover_girl)
        if c_slug in models_registry:
            models_registry[c_slug]["is_cover_girl"] = True
        else:
            models_registry[c_slug] = {
                "model_name": cover_girl,
                "slug": c_slug,
                "height": None,
                "weight_lbs": None,
                "measurements": None,
                "photographer": None,
                "pages": [0],
                "start_page": 0,
                "end_page": 0,
                "is_cover_girl": True,
                "bio_summary": f"Cover Girl for {meta['magazine_title']} {meta['issue_date']}"
            }

    doc.close()
    
    catalog = {
        "metadata": meta,
        "total_pages": total_pages,
        "cover_girl": cover_girl,
        "total_models_detected": len(models_registry),
        "models": models_registry
    }
    return catalog


if __name__ == "__main__":
    test_files = [
        "data/PB/Playboys_Girls_Of_Winter_1984.pdf",
        "data/PB/Playboys_Women_On_The_Move_1988.pdf",
        "data/PB/Playboys_Book_Of_Lingerie_1990-09_10.pdf",
        "data/PB/Playboys_Bathing_Beauties_1991.pdf",
        "data/PB/Playboys_Wet__Wild_1996.pdf",
        "data/PB/Playboys_Girls_Of_Summer_2000.pdf",
        "data/PB/Playboys_Vixens_2006-08_09.pdf",
        "data/PB/Playboys_Hot_Housewives_2008-09_10.pdf",
        "data/PB/Playboys_College_Girls_2009-01_02.pdf",
        "data/PB/Playboy_Germany_2011-01_Pdf_Xxx_Adult_Magazine.pdf",
        "data/PB/10playboy_Special_Collector_October-2014_Xxxmagazineporn.pdf",
        "data/PB/Playboy_Special_Collectors_Edition_01_2016_Pdf_Xxx_Adult_Magazine.pdf"
    ]
    
    print("=== TESTING GENERIFIED STAGE 1 ISSUE CATALOG EXTRACTOR ===\n")
    results = {}
    for f in test_files:
        if os.path.exists(f):
            cat = extract_issue_catalog(f)
            stem = cat["metadata"]["clean_stem"]
            results[stem] = cat
            print(f"[{cat['metadata']['year']}] {cat['metadata']['magazine_title']} ({cat['metadata']['issue_date']})")
            print(f"  Pages: {cat['total_pages']} | Cover Girl: {cat['cover_girl']} | Models Detected: {cat['total_models_detected']}")
            for mslug, minfo in list(cat['models'].items())[:4]:
                stats_str = f"Meas: {minfo['measurements']}" if minfo['measurements'] else "No stats"
                photo_str = f"Photo: {minfo['photographer']}" if minfo['photographer'] else "No photo credit"
                pg_str = f"P{minfo['start_page']}-P{minfo['end_page']}" if minfo['start_page'] is not None else "Pages: N/A"
                print(f"    • {minfo['model_name']:22s} | {pg_str:10s} | {stats_str:18s} | {photo_str}")
            print()
            
    # Save master index sample
    os.makedirs("data/eval", exist_ok=True)
    with open("data/visual_catalog_index_sample.json", "w", encoding="utf-8") as out:
        json.dump(results, out, indent=2)
    print("Saved sample index to data/visual_catalog_index_sample.json")
