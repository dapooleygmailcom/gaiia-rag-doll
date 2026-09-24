"""
Media & Visual Catalog Agent — Gaiia RAG Doll (Enhanced Multi-Modal Periodical Edition).

Retrieval agent for visual media, magazines, lookbooks, and image-heavy PDFs:
1. Translates natural language queries into semantic vectors and structured metadata filters
   (e.g., pose, hair color, vital stats, photographer, publication, decade, ads/casting).
2. Performs hybrid dual retrieval:
   - Exact Entity & Catalog lookup via 'data/visual_catalog_index.json' (O(1) model, spread, vital stats recall).
   - Semantic Vector Search across ChromaDB ('rag-doll-visual-catalog').
3. Formats high-fidelity visual cards with bodily dimensions, pose, photographer, styling,
   themes, and direct clickable links to rendered page images, facing spreads, and headshot thumbnails.
"""

import os
import re
import sys
import json
try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import ollama
except ImportError:
    ollama = None

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Script and Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data/chroma")
CHROMA_COLLECTION = "rag-doll-visual-catalog"
MASTER_INDEX_PATH = os.path.join(PROJECT_ROOT, "data/visual_catalog_index.json")


def get_chroma_collection():
    """Connect to the persistent ChromaDB collection."""
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(f"ChromaDB directory not found: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


def load_master_index():
    """Load the master exact lookup index if available."""
    if os.path.exists(MASTER_INDEX_PATH):
        try:
            with open(MASTER_INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"models": {}, "issues": {}, "photographers": {}}


def get_query_embedding(query_text):
    """Generate vector embedding for the query."""
    try:
        res = ollama.embeddings(model="nomic-embed-text", prompt=query_text)
        return res["embedding"]
    except Exception:
        try:
            res = ollama.embeddings(model="all-minilm", prompt=query_text)
            return res["embedding"]
        except Exception:
            return None


KNOWN_PHOTOGRAPHERS = [
    "Arny Freytag", "Stephen Wayda", "Richard Fegley", "Mizuno", "Gen Mishino",
    "Jarmo Pohjaniemi", "Byron Newman", "Ric Moore", "Sean Bolger", "J.R. Mounger",
    "Wesley Martens", "Sasha Eisenman", "Josh Ryan", "Autumn Sonnichsen", "David Mecey",
    "Kim Mizuno", "Mario Casilli", "Ken Marcus", "Guerin Blask", "Jeff Dunas"
]


QUERY_PARSER_PROMPT = """You are an expert visual catalog search assistant for magazines and pictorial publications.
Analyze the user's natural language search query and extract structured search parameters into JSON.

CRITICAL INSTRUCTION:
If a filter attribute is NOT explicitly requested or strongly required by the user query, set its value to null.
DO NOT hallucinate default values like 'Standing' for pose, 'Nature' for theme, or false for boolean fields!

JSON SCHEMA:
{{
  "semantic_search_text": "<Expanded search string optimized for embedding similarity>",
  "filters": {{
    "model_name": "<name if looking for a specific model or null>",
    "pose": "<'Standing' | 'Reclining' | 'Kneeling' | 'Sitting' | 'Arched_Back' | 'Close_Up' | 'All_Fours' | 'Lying_Down' | null>",
    "photographer": "<Photographer name if specified or null>",
    "hair_color": "<'Blonde' | 'Brunette' | 'Black' | 'Red' | null>",
    "primary_theme": "<'Beach' | 'Poolside' | 'Bedroom' | 'Studio' | 'Nature' | 'Glamour' | null>",
    "page_type": "<'Cover' | 'Feature_Pictorial' | 'Structured_Grid_Directory' | 'Advertisement' | 'Reader_Letters' | null>",
    "nudity_level": "<'Swimwear' | 'Lingerie' | 'Topless' | 'Full Nude' | 'Covered' | 'Artistic' | null>",
    "is_cover_girl": <true | null>,
    "was_playmate": <true | null>,
    "year": <integer year if mentioned or null>
  }}
}}

User query: "{query}"

Respond with ONLY the raw JSON object:"""


def slugify(text):
    """Generate safe lookup slug."""
    text = re.sub(r'[^\w\s-]', '', str(text).lower())
    return re.sub(r'[-\s]+', '_', text).strip('_')


LUXURY_FASHION_BRANDS = [
    "Givenchy", "Chanel", "Dior", "Gucci", "Prada", "Fendi", "Lanvin",
    "Saint Laurent", "Valentino", "Tom Ford", "Balmain", "Balenciaga",
    "Versace", "Armani", "Dolce & Gabbana", "Burberry", "Louis Vuitton",
    "Cartier", "Tiffany", "Chloé", "Max Mara", "Escada", "Coach",
    "Stuart Weitzman", "Harry Winston", "Alexander Wang", "Isabel Marant",
    "Agent Provocateur", "La Perla", "Calvin Klein", "Michael Kors", "Wolford"
]

GARMENT_SYNONYMS = {
    "scarves": "scarf", "scarf": "scarf", "turban": "turban", "turbans": "turban",
    "hat": "hat", "hats": "hat", "jacket": "jacket", "jackets": "jacket",
    "blazer": "blazer", "blazers": "blazer", "coat": "coat", "coats": "coat",
    "trench": "trench", "dress": "dress", "dresses": "dress", "gown": "gown", "gowns": "gown",
    "skirt": "skirt", "skirts": "skirt", "blouse": "blouse", "blouses": "blouse",
    "pants": "pants", "shoes": "shoes", "shoe": "shoes", "sandals": "sandals", "sandal": "sandals",
    "boots": "boots", "boot": "boots", "pumps": "pumps", "bag": "bag", "bags": "bag",
    "ring": "ring", "rings": "ring", "cuff": "cuff", "cuffs": "cuff",
    "lingerie": "lingerie", "swimwear": "swimwear", "bikini": "swimwear"
}


def extract_rule_based_filters(query):
    """Deterministic heuristic filter extractor for visual queries."""
    filters = {}
    query_lower = query.lower()
    
    # Year detection (single year)
    yr_match = re.search(r'\b(19\d\d|20\d\d)\b', query)
    if yr_match:
        filters["year"] = int(yr_match.group(1))
        
    # Decade detection
    decade_match = re.search(r'\b(19\d0s|20\d0s)\b', query_lower)
    if decade_match:
        dec_str = decade_match.group(1)
        dec_start = int(dec_str[:4])
        filters["decade_start"] = dec_start
        filters["decade_end"] = dec_start + 9
        
    # Pose detection (only if explicitly in query)
    for pose in ["standing", "reclining", "kneeling", "sitting", "arched_back", "close_up", "all_fours", "lying_down"]:
        if pose.replace("_", " ") in query_lower:
            filters["pose"] = pose.capitalize() if "_" not in pose else "Arched_Back" if "arched" in pose else "Close_Up"
            break
            
    # Hair detection
    for hair in ["blonde", "brunette", "black", "redhead", "auburn", "platinum"]:
        if hair in query_lower:
            filters["hair_color"] = "Red" if "red" in hair else hair.capitalize()
            break
            
    # Theme detection
    for theme in ["beach", "pool", "poolside", "bedroom", "studio", "outdoor", "nature", "retro"]:
        if theme in query_lower:
            filters["primary_theme"] = "Beach" if "beach" in theme else "Poolside" if "pool" in theme else theme.capitalize()
            break
            
    # Nudity level detection
    for nud in ["swimwear", "bikini", "lingerie", "topless", "full nude"]:
        if nud in query_lower:
            filters["nudity_level"] = "Swimwear" if nud in ["swimwear", "bikini"] else "Full Nude" if "full" in nud else nud.capitalize()
            break
            
    # Photographer detection (dynamic + known)
    photo_m = re.search(r'(?:shoots?\s+by|photos?\s+by|photographed\s+by|photography\s+by)\s+([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+)+)', query, re.IGNORECASE)
    if photo_m:
        filters["photographer"] = photo_m.group(1).strip()
    else:
        for photo in KNOWN_PHOTOGRAPHERS:
            last_name = photo.split()[-1].lower()
            if photo.lower() in query_lower or f"by {last_name}" in query_lower or f"photo {last_name}" in query_lower:
                filters["photographer"] = photo
                break

    # Wardrobe / Accessory item detection
    for term, canonical in GARMENT_SYNONYMS.items():
        if re.search(rf'\b{term}\b', query_lower):
            filters["wardrobe_item"] = canonical
            break

    # Designer / Brand detection
    for b in LUXURY_FASHION_BRANDS:
        if re.search(rf'\b{re.escape(b.lower())}\b', query_lower):
            filters["designer_brand"] = b
            break

    # Price range constraints
    max_m = re.search(r'(?:under|less\s+than|below)\s+[\$]?(\d[\d,]*(?:\.\d{2})?)', query_lower)
    if max_m:
        filters["price_max"] = float(max_m.group(1).replace(',', ''))
        
    min_m = re.search(r'(?:over|more\s+than|above)\s+[\$]?(\d[\d,]*(?:\.\d{2})?)', query_lower)
    if min_m:
        filters["price_min"] = float(min_m.group(1).replace(',', ''))
            
    # Model name detection heuristics (e.g. "featuring Carmella DeCesare", "Sara Stokes", "Julianne Moore")
    model_match = re.search(r'(?:featuring|model|spread of|pictorial of|photos of)\s+([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+)+)', query)
    if model_match:
        filters["model_name"] = model_match.group(1).strip()
            
    # Cover girl detection
    if "cover girl" in query_lower or "covergirl" in query_lower or "on the cover" in query_lower:
        filters["is_cover_girl"] = True
        
    return filters


def parse_query_intent(query):
    """Extract structured filters with robust rule-based baseline and null-safe LLM expansion."""
    rule_filters = extract_rule_based_filters(query)
    llm_filters = {}
    
    try:
        response = ollama.generate(
            model="gemma:2b",
            prompt=QUERY_PARSER_PROMPT.format(query=query),
            options={"temperature": 0.0}
        )
        content = response["response"].strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            parsed_json = json.loads(json_match.group())
            # Strip hallucinated default values
            for k, v in parsed_json.get("filters", {}).items():
                if v is not None and v is not False and v != "":
                    llm_filters[k] = v
    except Exception:
        pass

    # Merge: rule-based filters take absolute precedence
    merged_filters = {**llm_filters, **rule_filters}

    return {
        "semantic_search_text": query,
        "filters": merged_filters
    }


def parse_query_intent_with_profile(query, profile=None):
    """Parse query intent dynamically using domain profile schema and glossary."""
    return parse_query_intent(query)


class MediaAgent:
    """Agent for searching and browsing visual publication catalogs."""
    
    def __init__(self, profile_path=None):
        if profile_path:
            from engine.models.domain_profile import load_domain_profile
            self.profile = load_domain_profile(profile_path)
            self.collection_name = self.profile.chroma_collection
        else:
            self.profile = None
            self.collection_name = CHROMA_COLLECTION

        self.collection = get_chroma_collection()
        self.master_index = load_master_index()

    def format_retrieval_response(self, query, results):
        """Standardized interface for formatted retrieval results."""
        if isinstance(results, str):
            return results
        if isinstance(results, dict):
            return self.format_dual_results(query, results.get("exact", {}), results.get("vector"))
        return str(results)

    def search(self, query, top_k=5, strict_filters=False):
        """
        Execute dual exact catalog lookup across unified master index + semantic ChromaDB search.
        """
        parsed = parse_query_intent(query)
        search_text = parsed.get("semantic_search_text") or query
        filters = parsed.get("filters", {})
        query_lower = query.lower()
        
        exact_matches = {
            "models": [],
            "photographers": [],
            "designers": [],
            "accessories": []
        }
        
        # 1. Check Master Index for Model / Celebrity Name
        m_name = filters.get("model_name")
        if not m_name:
            for slug, minfo in self.master_index.get("models", {}).items():
                if minfo.get("model_name", "").lower() in query_lower:
                    m_name = minfo.get("model_name")
                    break
        if m_name:
            m_slug = slugify(m_name)
            if m_slug in self.master_index.get("models", {}):
                exact_matches["models"].append(self.master_index["models"][m_slug])

        # 2. Check Master Index for Photographer (Cross-genre!)
        p_name = filters.get("photographer")
        if not p_name:
            for slug, pinfo in self.master_index.get("photographers", {}).items():
                if pinfo.get("photographer", "").lower() in query_lower:
                    p_name = pinfo.get("photographer")
                    break
        if p_name:
            p_slug = slugify(p_name)
            if p_slug in self.master_index.get("photographers", {}):
                exact_matches["photographers"].append(self.master_index["photographers"][p_slug])

        # 3. Check Master Index for Designer / Brand
        d_name = filters.get("designer_brand")
        if not d_name:
            for slug, dinfo in self.master_index.get("designers_and_brands", {}).items():
                b_name = dinfo.get("brand_name", "")
                if len(b_name) >= 3 and re.search(rf'\b{re.escape(b_name.lower())}\b', query_lower):
                    d_name = b_name
                    break
        if d_name:
            d_slug = slugify(d_name)
            if d_slug in self.master_index.get("designers_and_brands", {}):
                exact_matches["designers"].append(self.master_index["designers_and_brands"][d_slug])

        # 4. Check Master Index for Wardrobe / Accessory Item
        w_item = filters.get("wardrobe_item")
        if not w_item:
            for item_key in self.master_index.get("wardrobe_accessories", {}):
                if re.search(rf'\b{re.escape(item_key.lower())}\b', query_lower):
                    w_item = item_key
                    break
        if w_item:
            w_slug = slugify(w_item)
            if w_slug in self.master_index.get("wardrobe_accessories", {}):
                items_list = self.master_index["wardrobe_accessories"][w_slug]
                exact_matches["accessories"].append({"item_name": w_item, "mentions": items_list})
                
        # 5. Build Chroma Where Clauses
        where_clauses = []
        for key, val in filters.items():
            if val is not None and val != "":
                if key == "hair_color":
                    where_clauses.append({"hair_color": {"$eq": str(val)}})
                elif key == "pose":
                    where_clauses.append({"pose": {"$eq": str(val)}})
                elif key == "year":
                    where_clauses.append({"year": {"$eq": int(val)}})
                elif key == "page_type":
                    where_clauses.append({"page_type": {"$eq": str(val)}})
                elif key == "primary_theme":
                    where_clauses.append({"primary_theme": {"$eq": str(val)}})
                elif key == "was_playmate":
                    where_clauses.append({"was_playmate": {"$eq": bool(val)}})
                elif key == "is_cover_girl":
                    where_clauses.append({"is_cover_girl": {"$eq": bool(val)}})
                elif key == "photographer":
                    where_clauses.append({"photographer": {"$eq": str(val)}})
                    
        where_arg = None
        if len(where_clauses) == 1:
            where_arg = where_clauses[0]
        elif len(where_clauses) > 1:
            where_arg = {"$and": where_clauses}

        query_emb = get_query_embedding(search_text)
        
        results = None
        query_kwargs = {
            "n_results": top_k,
            "include": ["metadatas", "documents", "distances"]
        }
        
        if query_emb:
            query_kwargs["query_embeddings"] = [query_emb]
        else:
            query_kwargs["query_texts"] = [search_text]
            
        if where_arg and strict_filters:
            query_kwargs["where"] = where_arg

        try:
            results = self.collection.query(**query_kwargs)
        except Exception:
            if where_arg:
                query_kwargs.pop("where", None)
                try:
                    results = self.collection.query(**query_kwargs)
                except Exception:
                    results = None

        return self.format_dual_results(query, exact_matches, results)

    def format_dual_results(self, query, exact_results, vector_results):
        """Format both exact index matches and semantic vector matches into rich markdown cards."""
        output = [f"## Visual Search Results for: \"{query}\"\n"]
        has_content = False

        if isinstance(exact_results, list):
            model_records = exact_results
            photog_records = []
            designer_records = []
            accessory_records = []
        else:
            model_records = exact_results.get("models", [])
            photog_records = exact_results.get("photographers", [])
            designer_records = exact_results.get("designers", [])
            accessory_records = exact_results.get("accessories", [])

        # 1. Format Model & Celebrity Matches
        if model_records:
            has_content = True
            for m_rec in model_records:
                m_name = m_rec.get("model_name")
                apps = m_rec.get("appearances", [])
                output.append(f"### ⭐ Authoritative Model Registry Match: **{m_name}** ({len(apps)} issue appearances)\n")
                for app in apps:
                    issue_title = app.get("magazine_title", "Publication")
                    year = app.get("year", "")
                    issue_date = app.get("issue_date", str(year))
                    pg_start = app.get("page_number", 0)
                    spread_pages = app.get("spread_pages", [pg_start])
                    pg_str = f"Pages {spread_pages[0]}–{spread_pages[-1]}" if len(spread_pages) > 1 else f"Page {pg_start}"
                    cover_badge = " 🌟 [Cover Girl]" if app.get("is_cover_girl") else ""
                    
                    attrs = app.get("physical_attributes", {})
                    meas = attrs.get("bodily_dimensions") or attrs.get("measurements") or "Unspecified"
                    height = attrs.get("height", "Unspecified")
                    hair = attrs.get("hair_color", "Unknown")
                    photog = app.get("photographer") or "Uncredited"
                    stylist = app.get("stylist")
                    wardrobe_items = app.get("wardrobe_items", [])
                    
                    thumb_path = app.get("thumbnail_path")
                    thumb_uri = thumb_path.replace("\\", "/") if thumb_path else None
                    
                    output.append(f"- **{issue_title}** ({issue_date}) — **{pg_str}**{cover_badge}")
                    if meas != "Unspecified" or height != "Unspecified":
                        output.append(f"  - **Vital Stats**: Measurements: `{meas}` | Height: `{height}` | Hair: `{hair}`")
                    if stylist:
                        output.append(f"  - ✨ **Stylist**: `{stylist}`")
                    if wardrobe_items:
                        w_desc = ", ".join(f"{w.get('designer', '')} {w.get('item', '')} (${w.get('price_usd'):.0f})" if w.get('price_usd') else f"{w.get('designer', '')} {w.get('item', '')}" for w in wardrobe_items)
                        output.append(f"  - 👗 **Wardrobe**: {w_desc}")
                    output.append(f"  - **Photography**: `{photog}`")
                    if thumb_uri:
                        output.append(f"  - 👤 **Thumbnail**: [{os.path.basename(thumb_path)}]({thumb_uri})")
                    output.append("")

        # 2. Format Photographer Matches
        if photog_records:
            has_content = True
            for p_rec in photog_records:
                p_name = p_rec.get("photographer")
                shoots = p_rec.get("shoots", [])
                output.append(f"### 📷 Photographer Portfolio: **{p_name}** ({len(shoots)} shoots across publications)\n")
                for shoot in shoots:
                    mag = shoot.get("magazine_title") or shoot.get("document_id", "Publication")
                    yr = f" ({shoot.get('year')})" if shoot.get("year") else ""
                    pg = shoot.get("page_number", "?")
                    sub = shoot.get("model_name") or shoot.get("editorial_title") or "Editorial Shoot"
                    output.append(f"- **{mag}**{yr} — Page {pg} — *{sub}*")
                output.append("")

        # 3. Format Designer Matches
        if designer_records:
            has_content = True
            for d_rec in designer_records:
                brand = d_rec.get("brand_name")
                mentions = d_rec.get("mentions", [])
                output.append(f"### 👗 Designer Catalog: **{brand}** ({len(mentions)} featured pieces)\n")
                for m in mentions:
                    price_str = f" (${m['price_usd']:.0f})" if m.get("price_usd") else ""
                    output.append(f"- **{m.get('document_id')}** — Page {m.get('page_number')}: {m.get('designer_line', brand)} **{m.get('item', '')}**{price_str}")
                output.append("")

        # 4. Format Wardrobe / Accessory Matches
        if accessory_records:
            has_content = True
            for a_rec in accessory_records:
                item_name = a_rec.get("item_name", "Item")
                mentions = a_rec.get("mentions", [])
                output.append(f"### 🧣 Wardrobe & Accessory Search: **{item_name.title()}** ({len(mentions)} pieces)\n")
                for m in mentions:
                    des_str = f" by {m['designer']}" if m.get('designer') else ""
                    price_str = f" (${m['price_usd']:.0f})" if m.get("price_usd") else ""
                    output.append(f"- **{m.get('document_id')}** — Page {m.get('page_number')}: **{m.get('item', item_name)}**{des_str}{price_str}")
                output.append("")

        # Format Semantic Vector Results from ChromaDB
        if vector_results and vector_results.get("ids") and len(vector_results["ids"][0]) > 0:
            has_content = True
            ids = vector_results["ids"][0]
            metadatas = vector_results["metadatas"][0]
            documents = vector_results["documents"][0]
            distances = vector_results.get("distances", [[]])[0]

            output.append(f"### 📸 Semantic & Visual Spreads ({len(ids)} matches)\n")

            for idx, (doc_id, meta, doc_text, dist) in enumerate(zip(ids, metadatas, documents, distances), 1):
                score = max(0.0, 1.0 - dist) if dist is not None else 1.0
                
                page_num = meta.get("page_number", "?")
                doc_name = meta.get("magazine_title") or meta.get("document_id", "Publication")
                issue_date = meta.get("issue_date", "")
                page_type = meta.get("page_type", "Page")
                model_name = meta.get("model_name", "Featured Subject")
                is_cover = meta.get("is_cover_girl", False)
                pose = meta.get("pose", "Unspecified")
                hair = meta.get("hair_color", "Unknown")
                height = meta.get("height", "Unspecified")
                dims = meta.get("bodily_dimensions", "Unspecified")
                nudity = meta.get("nudity_level", "Artistic")
                theme = meta.get("primary_theme", "Glamour")
                photographer = meta.get("photographer", "")
                tags = meta.get("tags", "")
                
                raw_img_path = meta.get("image_path", "")
                raw_spread_path = meta.get("spread_image_path", "")
                raw_thumb_path = meta.get("thumbnail_path", "")
                
                img_uri = raw_img_path.replace("\\", "/") if raw_img_path else "#"
                spread_uri = raw_spread_path.replace("\\", "/") if raw_spread_path else None
                thumb_uri = raw_thumb_path.replace("\\", "/") if raw_thumb_path else None
                
                cover_badge = " 🌟 [Cover Girl]" if is_cover else ""
                issue_label = f" ({issue_date})" if issue_date else ""
                
                output.append(f"#### {idx}. {doc_name}{issue_label} — Page {page_num} [{page_type}] (Match: {score:.1%})")
                output.append(f"- **Featured Subject**: **{model_name}**{cover_badge}")
                output.append(f"- **Pose & Styling**: Pose: `{pose}` | Nudity: `{nudity}`")
                output.append(f"- **Vital Stats & Physical**: Hair: `{hair}` | Height: `{height}` | Measurements: `{dims}`")
                if photographer:
                    output.append(f"- **Photography**: `{photographer}`")
                output.append(f"- **Setting & Theme**: `{theme}` | Tags: *{tags}*")
                output.append(f"- **Summary**: {doc_text[:250]}...")
                
                links_line = [f"🖼️ **Page**: [{os.path.basename(raw_img_path)}]({img_uri})"]
                if thumb_uri:
                    links_line.append(f"👤 **Headshot**: [{os.path.basename(raw_thumb_path)}]({thumb_uri})")
                if spread_uri:
                    links_line.append(f"📖 **2-Page Spread**: [{os.path.basename(raw_spread_path)}]({spread_uri})")
                output.append("- " + " | ".join(links_line))
                output.append("\n" + "-" * 50 + "\n")

        if not has_content:
            return f"### No visual records found matching: '{query}'"

        return "\n".join(output)


def interactive_cli():
    """Run an interactive CLI session with the MediaAgent."""
    agent = MediaAgent()
    print("\n" + "=" * 65)
    print(" 📸 Gaiia RAG Doll — Visual Media Retrieval Agent")
    print(" Type your search query below (e.g. 'models reclining on beach', 'shoots by Jarmo').")
    print(" Type 'exit' or 'quit' to end.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("\nVisual Query > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                break
                
            results = agent.search(query)
            print("\n" + results)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaiia RAG Doll Visual Media Retrieval Agent")
    parser.add_argument("--query", "-q", type=str, help="Search query to execute")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    agent = MediaAgent()
    if args.query:
        print(agent.search(args.query, top_k=args.top_k))
    else:
        interactive_cli()
