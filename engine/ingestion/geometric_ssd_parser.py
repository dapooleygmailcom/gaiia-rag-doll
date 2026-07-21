import os
import glob
import json
import re
import fitz  # PyMuPDF

def parse_all_pdfs(pdf_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    for pdf_path in pdf_files:
        try:
            entity = parse_single_pdf(pdf_path)
            if entity:
                basename = os.path.basename(pdf_path).replace(".pdf", "")
                out_path = os.path.join(out_dir, f"{basename}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(entity, f, indent=2)
                print(f"Successfully parsed {basename}")
        except Exception as e:
            print(f"Error parsing {pdf_path}: {e}")

def parse_single_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    
    spans = []
    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    text = s["text"].strip()
                    if text:
                        # Only take spans from the first vehicle on the page (y < 260)
                        if s["bbox"][3] < 265:
                            spans.append({
                                "text": text,
                                "x0": s["bbox"][0],
                                "y0": s["bbox"][1],
                                "x1": s["bbox"][2],
                                "y1": s["bbox"][3],
                                "xc": (s["bbox"][0] + s["bbox"][2]) / 2,
                                "yc": (s["bbox"][1] + s["bbox"][3]) / 2,
                            })
                            
    if not spans:
        return None
        
    entity = {
        "entity_type": "vehicle",
        "name": "Unknown",
        "attributes": {},
        "grids": {},
        "collections": {"Weapons": [], "Missiles": []}
    }
    
    # 1. Find Name and Type
    # Usually name is one of the larger texts, or near "Type:"
    type_span = next((s for s in spans if "Type:" in s["text"]), None)
    if type_span:
        # Find text to the right of "Type:"
        name_spans = [s for s in spans if s["yc"] > type_span["yc"] - 10 and s["yc"] < type_span["yc"] + 10 and s["x0"] > type_span["x1"]]
        if name_spans:
            name_spans.sort(key=lambda s: s["x0"])
            entity["name"] = name_spans[0]["text"]
            
    # 2. Maximum Thrust
    thrust_span = next((s for s in spans if "Maximum" in s["text"] or "Thrust" in s["text"] and s["x0"] > 500), None)
    if thrust_span:
        # Find number near it
        num_spans = [s for s in spans if s["x0"] > 530 and s["y0"] > 160 and s["text"].isdigit()]
        if num_spans:
            entity["attributes"]["Maximum Thrust"] = int(num_spans[0]["text"])

    # 3. Armor SFs
    facings = [
        ("Turret Armor", 60, 135),
        ("Stern Armor", 140, 215),
        ("Left Armor", 220, 295),
        ("Front Armor", 305, 380),
        ("Right Armor", 385, 460),
        ("Bottom Armor", 470, 545)
    ]
    
    sf_spans = [s for s in spans if s["text"].isdigit() and s["y0"] > 15 and s["y0"] < 35]
    for facing, x_min, x_max in facings:
        sf = next((int(s["text"]) for s in sf_spans if x_min <= s["xc"] <= x_max), 0)
        entity["grids"][facing] = {"SF": sf, "Width": 10, "Depth": sf} # Depth approx = SF for armor
        
    # 4. Weapons
    # Weapon headers are around y=204
    # "Weapon", "Loc.", "Dam.", "Rng."
    weapon_rows = [s for s in spans if s["y0"] > 215 and s["y0"] < 265]
    # Group by Y coordinate (rough rows)
    y_groups = {}
    for w in weapon_rows:
        y_key = round(w["yc"] / 5) * 5
        if y_key not in y_groups:
            y_groups[y_key] = []
        y_groups[y_key].append(w)
        
    for y_key, row_spans in y_groups.items():
        row_spans.sort(key=lambda s: s["x0"])
        # We need Name, Loc, Dam, Rng
        # X boundaries: Name < 100 or 180<x<250, Loc < 130 or 250<x<280, Dam < 160 or 280<x<315, Rng < 190 or 315<x<340
        # This is a heuristic matching
        if len(row_spans) >= 3:
            # First half of the page
            w1 = [s for s in row_spans if s["xc"] < 190]
            if len(w1) >= 3:
                name = w1[0]["text"]
                loc = w1[1]["text"]
                dam = w1[2]["text"]
                rng = w1[3]["text"] if len(w1) > 3 else "N/A"
                if not name.isdigit(): # Ignore if it's just numbers
                    entity["collections"]["Weapons"].append({"Name": name, "Location": loc, "Damage": dam, "Range": rng})
            # Second half of the page
            w2 = [s for s in row_spans if 190 < s["xc"] < 340]
            if len(w2) >= 3:
                name = w2[0]["text"]
                loc = w2[1]["text"]
                dam = w2[2]["text"]
                rng = w2[3]["text"] if len(w2) > 3 else "N/A"
                if not name.isdigit():
                    entity["collections"]["Weapons"].append({"Name": name, "Location": loc, "Damage": dam, "Range": rng})

    # 5. Missiles
    missile_names = [s for s in spans if s["x0"] < 60 and s["y0"] > 30 and s["y0"] < 150]
    # Combine SMLM / TVLG and "Missile"
    types_found = []
    for m in missile_names:
        t = m["text"].replace("Missile", "").strip()
        if t and t not in types_found:
            types_found.append(t)
            # Count boxes checked (heuristic: we just say Count=4 for TVLG, Count=1 for SMLM as fallback)
            # In a real parser we'd count the marked boxes. For now:
            entity["collections"]["Missiles"].append({"Type": t, "Count": 4 if "TVLG" in t else 1})

    # 6. Internal Grids
    # We will map text components found between y=110 and y=200
    internal_spans = [s for s in spans if s["yc"] > 115 and s["yc"] < 200 and not s["text"].isdigit()]
    
    def build_grid(x_min, x_max, width):
        cols = [[] for _ in range(width)]
        col_width = (x_max - x_min) / width
        block_spans = [s for s in internal_spans if x_min - 10 <= s["xc"] <= x_max + 10]
        # Sort by Y to process top-to-bottom
        block_spans.sort(key=lambda s: s["yc"])
        
        for s in block_spans:
            # Determine which columns this span covers
            c_start = max(0, int((s["x0"] - x_min) / col_width))
            c_end = min(width - 1, int((s["x1"] - x_min) / col_width))
            # Fallback if x0/x1 are a bit off
            if c_end < c_start: c_end = c_start
            
            for c in range(c_start, c_end + 1):
                cols[c].append(s["text"])
                
        # Pad to 10 depth
        for c in range(width):
            while len(cols[c]) < 10:
                cols[c].append("Empty")
            # Truncate if more than 10
            cols[c] = cols[c][:10]
        return cols

    # Turret Internals (X: 60 - 135)
    entity["grids"]["Turret Internals"] = {"Width": 10, "Columns": build_grid(60, 130, 10)}
    
    # Bottom Internals (X: 470 - 545)
    entity["grids"]["Bottom Internals"] = {"Width": 10, "Columns": build_grid(470, 540, 10)}
    
    # Main Internals (46 columns: Stern, Gap, Left, Gap, Front, Gap, Right)
    # Stern: 140-210, Left: 220-290, Front: 300-370, Right: 380-450
    # Gaps are ~10 pixels wide each. Total width 140 to 450 = 310 pixels. 310 / 46 = 6.74 pixels/col
    entity["grids"]["Main Internals"] = {"Width": 46, "Columns": build_grid(140, 450, 46)}
    
    return entity

if __name__ == "__main__":
    import sys
    pdf_dir = "data/rl"
    out_dir = "data/entities"
    if len(sys.argv) > 1:
        pdf_dir = sys.argv[1]
    if len(sys.argv) > 2:
        out_dir = sys.argv[2]
    parse_all_pdfs(pdf_dir, out_dir)
