"""
Media & Visual Catalog Agent — Gaiia RAG Doll (Enhanced Multi-Modal Periodical Edition).

Retrieval agent for visual media, magazines, lookbooks, and image-heavy PDFs:
1. Translates natural language queries into semantic vectors and structured metadata filters
   (e.g., pose, hair color, vital stats, photographer, publication, ads/casting).
2. Performs hybrid vector search across ChromaDB ('rag-doll-visual-catalog').
3. Formats high-fidelity visual cards with bodily dimensions, pose, photographer, styling,
   themes, and direct clickable links to rendered page images, facing spreads, and headshot thumbnails.
"""

import os
import re
import sys
import json
import argparse
import chromadb
import ollama

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

def get_chroma_collection():
    """Connect to the persistent ChromaDB collection."""
    if not os.path.exists(CHROMA_DB_DIR):
        raise FileNotFoundError(f"ChromaDB directory not found: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


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


QUERY_PARSER_PROMPT = """You are an expert visual catalog search assistant for magazines and pictorial publications.
Analyze the user's natural language search query and extract structured search parameters into JSON.

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
    "is_cover_girl": <true | false | null>,
    "was_playmate": <true | false | null>,
    "year": <integer year if mentioned or null>
  }}
}}

User query: "{query}"

Respond with ONLY the raw JSON object:"""


def extract_rule_based_filters(query):
    """Deterministic heuristic filter extractor for visual queries."""
    filters = {}
    query_lower = query.lower()
    
    # Year detection
    yr_match = re.search(r'\b(19\d\d|20\d\d)\b', query)
    if yr_match:
        filters["year"] = int(yr_match.group(1))
        
    # Pose detection
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
            
    # Photographer detection
    for photo in ["pohjaniemi", "jarmo", "mishino", "gen", "mizuno", "newman", "byron", "moore", "ric"]:
        if photo in query_lower:
            if photo in ["pohjaniemi", "jarmo"]:
                filters["photographer"] = "Jarmo Pohjaniemi"
            elif photo in ["mishino", "gen"]:
                filters["photographer"] = "Gen Mishino"
            elif photo in ["mizuno"]:
                filters["photographer"] = "Mizuno"
            elif photo in ["newman", "byron"]:
                filters["photographer"] = "Byron Newman"
            elif photo in ["moore", "ric"]:
                filters["photographer"] = "Ric Moore"
            break
            
    # Cover girl detection
    if "cover girl" in query_lower or "covergirl" in query_lower or "on the cover" in query_lower:
        filters["is_cover_girl"] = True
        
    return filters


def parse_query_intent(query):
    """Extract structured filters with robust rule-based baseline and optional LLM expansion."""
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
            llm_filters = {k: v for k, v in parsed_json.get("filters", {}).items() if v is not None}
    except Exception:
        pass

    # Merge: rule-based filters take precedence for explicit keywords, LLM fills in the rest
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

    def search(self, query, top_k=5, strict_filters=False):
        """
        Execute semantic + metadata filtered search.
        """
        parsed = parse_query_intent(query)
        search_text = parsed.get("semantic_search_text") or query
        filters = parsed.get("filters", {})
        
        # Build Chroma where clause
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
        except Exception as e:
            if where_arg:
                query_kwargs.pop("where", None)
                results = self.collection.query(**query_kwargs)
            else:
                raise e

        return self.format_results(query, results)

    def format_results(self, query, results):
        """Format ChromaDB results into rich, readable visual attribute cards with thumbnails and links."""
        if not results or not results.get("ids") or len(results["ids"][0]) == 0:
            return f"### No visual records found matching: '{query}'"

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]
        distances = results.get("distances", [[]])[0]

        output = [f"## Visual Search Results for: \"{query}\"\n"]
        output.append(f"*Found {len(ids)} matching records in visual catalog:*\n")

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
            
            # Format file URI for Windows clickable links
            img_uri = "file:///" + raw_img_path.replace("\\", "/") if raw_img_path else "#"
            spread_uri = "file:///" + raw_spread_path.replace("\\", "/") if raw_spread_path else None
            thumb_uri = "file:///" + raw_thumb_path.replace("\\", "/") if raw_thumb_path else None
            
            cover_badge = " 🌟 [Cover Girl]" if is_cover else ""
            issue_label = f" ({issue_date})" if issue_date else ""
            
            output.append(f"### {idx}. {doc_name}{issue_label} — Page {page_num} [{page_type}] (Match: {score:.1%})")
            output.append(f"- **Featured Subject**: **{model_name}**{cover_badge}")
            output.append(f"- **Pose & Styling**: Pose: `{pose}` | Nudity: `{nudity}`")
            output.append(f"- **Vital Stats & Physical**: Hair: `{hair}` | Height: `{height}` | Measurements: `{dims}`")
            if photographer:
                output.append(f"- **Photography**: `{photographer}`")
            output.append(f"- **Setting & Theme**: `{theme}` | Tags: *{tags}*")
            output.append(f"- **Context & Summary**: {doc_text[:280]}...")
            
            links_line = [f"🖼️ **Page**: [{os.path.basename(raw_img_path)}]({img_uri})"]
            if thumb_uri:
                links_line.append(f"👤 **Headshot**: [{os.path.basename(raw_thumb_path)}]({thumb_uri})")
            if spread_uri:
                links_line.append(f"📖 **2-Page Spread**: [{os.path.basename(raw_spread_path)}]({spread_uri})")
            output.append("- " + " | ".join(links_line))
            output.append("\n" + "-" * 60 + "\n")

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
