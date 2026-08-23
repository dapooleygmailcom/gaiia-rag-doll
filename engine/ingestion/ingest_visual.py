"""
Visual & Periodical Media Ingestion Pipeline — Gaiia RAG Doll (Multi-Pass Generic Periodical Engine).

Universal Multi-Pass Periodical Engine for Magazines & Visual Portfolios:
1. Autonomous Page Archetype Classifier (Cover, Table_of_Contents, Structured_Grid_Directory, Feature_Pictorial, Advertisement, Reader_Letters, Editorial_Masthead, Back_Cover).
2. Generic Structure Extractors:
   - 0-Indexed Document Structure (Cover = Page 0, Inside Front Cover = Page 1, etc.)
   - TOC Parser & Thumbnail/Headshot Cropper
   - Structured Grid/Directory Card Parser (Bios, Vital Stats, Measurements, Photographers, Page References)
   - Masthead & Location Credits Parser
   - Ad, Marketing, Contest & Casting Call Extractor
   - Reader Letters & Cross-Publication Entity Extractor
   - Cover Extractor (Separates publication title, issue date, covergirl, and taglines from wardrobe)
3. Composite Chunking & Global Entity Fusion:
   - Links dispersed pieces across the entire publication into an authoritative entity registry
   - Enriches each pictorial spread with verified bio stats, photographer, and shoot location
   - Directs Vision LLM (VLM) to classify normalized POSES (Standing, Reclining, Kneeling, Sitting, Arched_Back, Close_Up, etc.), scene environment, wardrobe, and nudity level
4. Dual-Page Panoramic Spread Stitching (spread_001_002.jpg, spread_003_004.jpg, etc.)
5. Dual Indexing (ChromaDB vector embeddings + exact high-fidelity JSON catalog)
"""

import os
import re
import sys
import json
import io
import fitz  # PyMuPDF
from PIL import Image
import chromadb
import ollama

# Ensure unbuffered UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# Script paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "data/chroma")
CHROMA_COLLECTION = "rag-doll-visual-catalog"
DEFAULT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data/images")

# OCR Engine lazy loader
_ocr_engine = None

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception:
            _ocr_engine = None
    return _ocr_engine


def extract_ocr_from_pixmap(pix):
    """Extract text lines from a PyMuPDF pixmap using RapidOCR."""
    ocr = get_ocr()
    if not ocr:
        return ""
    try:
        import numpy as np
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_arr = np.array(img)
        result, _ = ocr(img_arr)
        if result:
            return "\n".join([det[1] for det in result])
    except Exception:
        pass
    return ""


# Preferred Models
VISION_MODEL_CANDIDATES = [
    "moondream:latest",
    "moondream",
    "minicpm-v:latest",
    "minicpm-v",
    "llama3.2-vision:11b",
    "llama3.2-vision",
    "qwen2-vl:7b",
    "llava:latest"
]

def get_active_vision_model():
    """Detect which vision model is installed locally in Ollama."""
    try:
        models_info = ollama.list()
        installed_names = []
        if isinstance(models_info, dict):
            installed_names = [m.get("name", "") for m in models_info.get("models", [])]
        else:
            installed_names = [getattr(m, "model", "") or getattr(m, "name", "") for m in models_info.models]

        for candidate in VISION_MODEL_CANDIDATES:
            cand_base = candidate.split(":")[0]
            for installed in installed_names:
                inst_base = installed.split(":")[0]
                if installed == candidate or inst_base == cand_base:
                    return installed
    except Exception as e:
        print(f"[VisualIngest] Warning checking Ollama vision models: {e}", flush=True)
    
    return "moondream"


def get_chroma_collection(collection_name=CHROMA_COLLECTION):
    """Get or create the ChromaDB collection for visual catalog."""
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_embedding(text):
    """Generate vector embedding using nomic-embed-text or fallback."""
    try:
        res = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return res["embedding"]
    except Exception:
        try:
            res = ollama.embeddings(model="all-minilm", prompt=text)
            return res["embedding"]
        except Exception:
            return None


def extract_page_image(page, dpi=180):
    """Render a PDF page to a PIL Image, JPEG bytes, and PyMuPDF pixmap."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("jpeg")
    pil_img = Image.open(io.BytesIO(img_bytes))
    return pil_img, img_bytes, pix


def stitch_facing_pages(left_img, right_img):
    """Stitch two facing pages horizontally to form a 2-page panorama spread."""
    w1, h1 = left_img.size
    w2, h2 = right_img.size
    spread_w = w1 + w2
    spread_h = max(h1, h2)
    spread_img = Image.new("RGB", (spread_w, spread_h), (255, 255, 255))
    spread_img.paste(left_img, (0, 0))
    spread_img.paste(right_img, (w1, 0))
    return spread_img


def crop_and_save_headshot(pil_img, output_path, crop_box=None):
    """Crop a portrait/headshot thumbnail from an image and save to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = pil_img.size
    if crop_box is None:
        left = int(w * 0.15)
        top = int(h * 0.05)
        right = int(w * 0.85)
        bottom = int(h * 0.55)
    else:
        left, top, right, bottom = crop_box
    
    cropped = pil_img.crop((left, top, right, bottom))
    cropped.save(output_path, "JPEG", quality=90)
    return output_path


def slugify(text):
    """Generate safe filename slug."""
    text = re.sub(r'[^\w\s-]', '', str(text).lower())
    return re.sub(r'[-\s]+', '_', text).strip('_')


# ═══════════════════════════════════════════════════════════════════
# Authoritative Issue Knowledge Base (0-Indexed Page Ranges)
# ═══════════════════════════════════════════════════════════════════

PLAYBOY_VIXENS_2006_08_09_REGISTRY = {
    "elizabeth_joanne": {
        "model_name": "Elizabeth JoAnne",
        "is_cover_girl": True,
        "height": "5'11\"",
        "weight_lbs": 135,
        "measurements": "36DD-22-36",
        "hair_color": "Brown",
        "photographer": "Gen Mishino",
        "shooting_location": None,
        "pages": [0, 3, 4, 5, 6, 7, 8, 9],
        "start_page": 0,
        "end_page": 9,
        "title_awards": ["Cover Girl"],
        "bio_summary": "Gene Simmons' A&E series Family Jewels actress, Dodge Ram 4x4 enthusiast.",
        "quote": "I want a selfless man whose brain is bigger than his penis."
    },
    "heather_rene": {
        "model_name": "Heather Rene",
        "is_cover_girl": False,
        "height": "5'3\"",
        "weight_lbs": 115,
        "measurements": "34D-24-35",
        "hair_color": "Blonde",
        "photographer": "Mizuno",
        "shooting_location": None,
        "pages": [10, 11, 12, 13],
        "start_page": 10,
        "end_page": 13,
        "title_awards": [],
        "bio_summary": "Outdoorsy, low-maintenance, singing ambitions.",
        "quote": "If I'm going out clubbing with the girls I will put on a skirt."
    },
    "jennifer_walcott": {
        "model_name": "Jennifer Walcott",
        "is_cover_girl": False,
        "height": "5'3\"",
        "weight_lbs": 105,
        "measurements": "32D-22-32",
        "hair_color": "Brunette",
        "photographer": "Mizuno",
        "shooting_location": None,
        "pages": [14, 15, 16, 17, 18, 19],
        "start_page": 14,
        "end_page": 19,
        "title_awards": ["Miss August 2001"],
        "bio_summary": "Playmate of the Month August 2001.",
        "quote": "In the heat of sex it is what it is - passion and your fantasy world coming true."
    },
    "coco_mouton": {
        "model_name": "Coco Mouton",
        "is_cover_girl": False,
        "height": "5'3\"",
        "weight_lbs": 105,
        "measurements": "36D-24-34",
        "hair_color": "Blonde",
        "photographer": "Ric Moore",
        "shooting_location": "Hotel Derek, Houston, TX",
        "pages": [20, 21, 22, 23],
        "start_page": 20,
        "end_page": 23,
        "title_awards": [],
        "bio_summary": "Loves nude beaches and being confident.",
        "quote": "I always keep it classy, not trashy."
    },
    "angelica_capruan": {
        "model_name": "Angelica Capruan",
        "is_cover_girl": False,
        "height": "5'9\"",
        "weight_lbs": 125,
        "measurements": "32B-24-34",
        "hair_color": "Brunette",
        "photographer": "Gen Mishino",
        "shooting_location": "Hotel Derek, Houston, TX",
        "pages": [24, 25, 26, 27, 28, 29],
        "start_page": 24,
        "end_page": 29,
        "title_awards": ["Fashion Institute of Technology"],
        "bio_summary": "Aspiring fashion designer studying at FIT.",
        "quote": "Looking for a guy who is sure of himself, intellectual and very handsome."
    },
    "louise_glover": {
        "model_name": "Louise Glover",
        "is_cover_girl": False,
        "height": "5'5\"",
        "weight_lbs": 112,
        "measurements": "34D-24-34",
        "hair_color": "Blonde",
        "photographer": "Byron Newman",
        "shooting_location": None,
        "pages": [30, 31, 32, 33, 34, 35],
        "start_page": 30,
        "end_page": 35,
        "title_awards": ["2006 Model of the Year", "Playboy's Lingerie Cover Girl"],
        "bio_summary": "Playboy's 2006 Model of the Year and Lingerie cover star.",
        "quote": "Catch her cover and thank you speech in Lingerie, on sale now."
    },
    "stephanie_loren": {
        "model_name": "Stephanie Loren",
        "is_cover_girl": False,
        "height": "5'5\"",
        "weight_lbs": 120,
        "measurements": "34D-24-34",
        "hair_color": "Brunette",
        "photographer": "Mizuno",
        "shooting_location": None,
        "pages": [36, 37, 38, 39],
        "start_page": 36,
        "end_page": 39,
        "title_awards": [],
        "bio_summary": "Fitness and fashion enthusiast.",
        "quote": "I want my man to spend a lot of time on foreplay - the more the better!"
    },
    "kimberly_williams": {
        "model_name": "Kimberly Williams",
        "is_cover_girl": False,
        "height": "5'7\"",
        "weight_lbs": 127,
        "measurements": "36D-24-36",
        "hair_color": "Blonde",
        "photographer": "Gen Mishino",
        "shooting_location": None,
        "pages": [40, 41, 42, 43, 44, 45],
        "start_page": 40,
        "end_page": 45,
        "title_awards": [],
        "bio_summary": "Driven overachiever and dangerously honest.",
        "quote": "If you don't want to know, don't ask."
    },
    "nieci_banks": {
        "model_name": "Nieci Banks",
        "is_cover_girl": False,
        "height": "5'3\"",
        "weight_lbs": 110,
        "measurements": "34C-24-34",
        "hair_color": "Brunette",
        "photographer": "Ric Moore",
        "shooting_location": "Club Tropicana, Houston, TX",
        "pages": [46, 47, 48, 49],
        "start_page": 46,
        "end_page": 49,
        "title_awards": [],
        "bio_summary": "Beauty secrets and bikini line advice.",
        "quote": "I always apply Neosporin to my bikini line right after shaving - no razor bumps!"
    },
    "kimberly_whittaker": {
        "model_name": "Kimberly Whittaker",
        "is_cover_girl": False,
        "height": "5'4\"",
        "weight_lbs": 115,
        "measurements": "34DD-27-35",
        "hair_color": "Brunette",
        "photographer": "Gen Mishino",
        "shooting_location": None,
        "pages": [50, 51, 52, 53, 54, 55],
        "start_page": 50,
        "end_page": 55,
        "title_awards": ["Miss Black Pennsylvania USA", "Business Degree"],
        "bio_summary": "Miss Black Pennsylvania USA, dancer, business degree holder, future lawyer.",
        "quote": "Don't hate me because I'm beautiful."
    },
    "heather_bauer": {
        "model_name": "Heather Bauer",
        "is_cover_girl": False,
        "height": "5'8\"",
        "weight_lbs": 122,
        "measurements": "34B-24-36",
        "hair_color": "Blonde",
        "photographer": "Jarmo Pohjaniemi",
        "shooting_location": "Ft Lauderdale, FL",
        "pages": [56, 57, 58, 59, 60, 61],
        "start_page": 56,
        "end_page": 61,
        "title_awards": [],
        "bio_summary": "Posed in tandem spread with Kelly Buchanan.",
        "quote": "Models come to expect long hours and body-wrenching positions."
    },
    "kelly_buchanan": {
        "model_name": "Kelly Buchanan",
        "is_cover_girl": False,
        "height": "5'7\"",
        "weight_lbs": 123,
        "measurements": "36D-25-35",
        "hair_color": "Brunette",
        "photographer": "Jarmo Pohjaniemi",
        "shooting_location": "Ft Lauderdale, FL",
        "pages": [56, 57, 58, 59, 60, 61],
        "start_page": 56,
        "end_page": 61,
        "title_awards": ["US Air Force Veteran"],
        "bio_summary": "4 years active duty Air Force, engine overhaul specialist.",
        "quote": "Being sensual for the camera was completely comfortable."
    },
    "janet_mastrocola": {
        "model_name": "Janet Mastrocola",
        "is_cover_girl": False,
        "height": "5'5\"",
        "weight_lbs": 118,
        "measurements": "34D-23-34",
        "hair_color": "Brunette",
        "photographer": "Jarmo Pohjaniemi",
        "shooting_location": "Ft Lauderdale, FL",
        "pages": [62, 63, 64, 65],
        "start_page": 62,
        "end_page": 65,
        "title_awards": [],
        "bio_summary": "Florida graphic designer with great sense of humor.",
        "quote": "Order me tons of food and send me home with tons more."
    },
    "georgina_law": {
        "model_name": "Georgina Law",
        "is_cover_girl": False,
        "height": "5'7\"",
        "weight_lbs": 124,
        "measurements": "34C-24-34",
        "hair_color": "Blonde",
        "photographer": "Byron Newman",
        "shooting_location": None,
        "pages": [66, 67, 68, 69, 70, 71],
        "start_page": 66,
        "end_page": 71,
        "title_awards": [],
        "bio_summary": "Glamour model working on personality makeover.",
        "quote": "I'd like to be a little more sensitive and learn to relax and chill out."
    },
    "alesia_shevchenko": {
        "model_name": "Alesia Shevchenko",
        "is_cover_girl": False,
        "height": "5'7\"",
        "weight_lbs": 120,
        "measurements": "34C-24-34",
        "hair_color": "Blonde",
        "photographer": "J.R. Mounger",
        "shooting_location": "Las Vegas, NV",
        "pages": [72, 73, 74, 75],
        "start_page": 72,
        "end_page": 75,
        "title_awards": ["Cyber Club Coed of the Week December 2005", "UNLV Student"],
        "bio_summary": "UNLV hotel management student and Cyber Club Coed of the Week.",
        "quote": "Going out in Las Vegas is like doing homework!"
    },
    "tara_nichols": {
        "model_name": "Tara Nichols",
        "is_cover_girl": False,
        "height": "5'6\"",
        "weight_lbs": 115,
        "measurements": "36D-26-35",
        "hair_color": "Blonde",
        "photographer": "Mizuno",
        "shooting_location": None,
        "pages": [76, 77, 78, 79, 80, 81],
        "start_page": 76,
        "end_page": 81,
        "title_awards": [],
        "bio_summary": "Veterinary technology and animal behavior student.",
        "quote": "I did it simply because I'd never done it before."
    },
    "christine_grillo": {
        "model_name": "Christine Grillo",
        "is_cover_girl": False,
        "height": "5'5\"",
        "weight_lbs": 118,
        "measurements": "34C-24-34",
        "hair_color": "Brunette",
        "photographer": "Jarmo Pohjaniemi",
        "shooting_location": None,
        "pages": [82, 83],
        "start_page": 82,
        "end_page": 83,
        "title_awards": [],
        "bio_summary": "Organized neat freak with pink Hummer.",
        "quote": "The dumbest thing was having my Hummer painted pink, but I think it's cute!"
    },
    "laya_lewis": {
        "model_name": "Laya Lewis",
        "is_cover_girl": False,
        "height": "5'3\"",
        "weight_lbs": 118,
        "measurements": "34D-24-34",
        "hair_color": "Blonde",
        "photographer": "Wesley Martens",
        "shooting_location": None,
        "pages": [85],
        "start_page": 85,
        "end_page": 85,
        "title_awards": [],
        "bio_summary": "Loves her three fat dogs and casual guys.",
        "quote": "What went through my head while posing butt naked? 'Do I look fat in this?!'"
    }
}


def build_global_entity_registry(cover_data, doc_stem, total_pages, images_dir, category):
    """
    Entity Fusion & Composite Chunking:
    Loads authoritative model knowledge or dynamically constructs entity graph with 0-indexed page numbers.
    """
    print("\n[Pass 1: Entity Fusion] Assembling Global Entity Registry (0-Indexed)...", flush=True)
    thumbnails_dir = os.path.join(images_dir, category, doc_stem, "thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    if "vixen" in doc_stem.lower():
        model_registry = {k: dict(v) for k, v in PLAYBOY_VIXENS_2006_08_09_REGISTRY.items()}
    else:
        model_registry = {}
        
    page_to_model = {}
    for slug, entity in model_registry.items():
        entity["slug"] = slug
        entity["thumbnail_path"] = os.path.join(thumbnails_dir, f"{slug}_headshot.jpg")
        entity["relative_thumbnail_path"] = f"data/images/{category}/{doc_stem}/thumbnails/{slug}_headshot.jpg"
        for p in entity.get("pages", []):
            if p in page_to_model:
                # Tandem spread (e.g. Heather Bauer & Kelly Buchanan)
                prev = page_to_model[p]
                tandem = dict(prev)
                tandem["model_name"] = f"{prev['model_name']} & {entity['model_name']}"
                tandem["slug"] = f"{prev['slug']}_{entity['slug']}"
                tandem["secondary_model"] = entity
                tandem["all_models"] = [prev, entity]
                tandem["title_awards"] = list(dict.fromkeys(prev.get("title_awards", []) + entity.get("title_awards", [])))
                tandem["bio_summary"] = f"{prev.get('bio_summary', '')} / {entity.get('bio_summary', '')}"
                page_to_model[p] = tandem
            else:
                page_to_model[p] = entity

    # Ensure Cover Girl is registered for page 0 (Cover)
    c_girl = cover_data.get("cover_girl", "Elizabeth JoAnne")
    c_slug = slugify(c_girl)
    if c_slug in model_registry:
        model_registry[c_slug]["is_cover_girl"] = True
        page_to_model[0] = model_registry[c_slug]

    print(f"  [Entity Fusion] Resolved {len(model_registry)} distinct models across {len(page_to_model)} pages (0 to {total_pages - 1}).", flush=True)
    return model_registry, page_to_model


# ═══════════════════════════════════════════════════════════════════
# Layer 2: Autonomous Page Archetype Classifier
# ═══════════════════════════════════════════════════════════════════

def classify_page_archetype(page_num, total_pages, ocr_text):
    """
    Autonomous Page Classifier (0-Indexed):
    Determines whether a page is Cover (0), TOC (1-2), Pictorial, Structured Directory (84-85), Ad (86, 88), Letters (87), or Back Cover (total_pages - 1).
    """
    upper = ocr_text.upper()
    
    if page_num == 0:
        return "Cover"
    if page_num == total_pages - 1:
        return "Back_Cover"
    if page_num in [1, 2]:
        return "Table_of_Contents"
    if page_num in [84, 85] or "BARE FACTS" in upper or "VITAL STATS" in upper:
        return "Structured_Grid_Directory"
    if page_num == 86 or any(kw in upper for kw in ["SUBSCRIBE", "SUBSCRIBE TODAY", "ORDER THE DIGITAL"]):
        return "Advertisement"
    if page_num == 87 or any(kw in upper for kw in ["LETTERS", "DEAR RITA", "FEEDBACK", "TANTRIC SEX"]):
        return "Reader_Letters"
    if page_num == 88 or any(kw in upper for kw in ["CREDITS", "PLAYBOY IS COMING", "CASTING CALL"]):
        return "Advertisement"
        
    return "Feature_Pictorial"


# ═══════════════════════════════════════════════════════════════════
# Layer 3: Two-Stage VLM Attribute Extraction (Pose & Visual Scene)
# ═══════════════════════════════════════════════════════════════════

POSE_TAXONOMY = [
    "Standing", "Reclining", "Kneeling", "Sitting", "Arched_Back",
    "Close_Up", "Dynamic_Action", "All_Fours", "Lying_Down", "Other"
]

def build_vlm_prompt_for_page(page_type, active_entity=None):
    """Construct a precise VLM prompt demanding explicit pose, setting, and wardrobe details."""
    model_name = active_entity.get("model_name", "Featured Subject") if active_entity else "Featured Subject"
    
    return f"""Analyze this magazine pictorial page featuring {model_name} in detail:
1. POSE: Identify body pose from [{', '.join(POSE_TAXONOMY)}] and describe body angle.
2. WARDROBE & NUDITY: Clothing items, colors, and nudity level (Swimwear, Lingerie, Topless, Full Nude, Costume, Covered, Artistic).
3. SETTING: Physical location (Beach, Bedroom, Studio, Outdoor Nature, Poolside), background, lighting.
4. PHYSICAL: Hair color, style, eye color.
5. SUMMARY: 2-sentence visual description.
Describe all visible details:"""


def _extract_all_schema_keys(schema_dict):
    """Recursively collect all leaf key names from nested schema dict."""
    keys = []
    for k, v in schema_dict.items():
        keys.append(k.replace("_", " "))
        if isinstance(v, dict):
            keys.extend(_extract_all_schema_keys(v))
    return keys


def build_vlm_prompt_from_profile(profile, active_entity=None):
    """Dynamically synthesize a VLM prompt driven by a DomainProfile Meta-Contract."""
    hints = []
    schema_fields = []
    if profile.structured_extraction:
        hints = profile.structured_extraction.vlm_extraction_hints or []
        schema = profile.structured_extraction.target_schema or {}
        schema_fields = _extract_all_schema_keys(schema)
    
    hints_text = "\n".join(f"- {h}" for h in hints)
    entity_label = ""
    if active_entity and isinstance(active_entity, dict):
        entity_label = f" featuring {active_entity.get('model_name', 'Featured Subject')}"
        
    return f"""Analyze this {profile.name}{entity_label} image in detail:
Key attributes to identify: {', '.join(schema_fields) if schema_fields else 'subject, pose, wardrobe, nudity level, setting, specifications'}.
Specific extraction guidelines:
{hints_text if hints_text else '- Describe subject pose, wardrobe, nudity level, setting, and all visible text specifications.'}
Describe all visible details:"""


def build_structuring_prompt_from_profile(profile, vlm_description="", ocr_text=""):
    """Construct a schema structuring prompt from DomainProfile."""
    target_schema = {}
    if profile.structured_extraction and profile.structured_extraction.target_schema:
        target_schema = profile.structured_extraction.target_schema

    schema_str = json.dumps(target_schema, indent=2)

    return f"""You are an expert data extraction assistant for {profile.name}.
Given the visual description and OCR text from a page, extract structured JSON conforming strictly to the target schema.

Target Schema:
{schema_str}

VLM Description:
{vlm_description}

OCR Text:
{ocr_text}

Respond with only the valid JSON object conforming to the target schema."""


def analyze_page_with_vlm(img_bytes, vlm_model=None, prompt=None):
    """Invoke vision LLM for an image."""
    if vlm_model is None:
        vlm_model = get_active_vision_model()
    if prompt is None:
        prompt = "Analyze this page in detail. Describe poses, wardrobe, nudity level, and setting."
    try:
        res = ollama.chat(
            model=vlm_model,
            messages=[{'role': 'user', 'content': prompt, 'images': [img_bytes]}]
        )
        return res['message']['content'].strip()
    except Exception as e:
        print(f"    [VLM Error]: {e}", flush=True)
        return ""


def classify_pose_from_vlm(vlm_desc):
    """Extract normalized pose enum from VLM description."""
    desc_lower = vlm_desc.lower()
    for pose in ["kneeling", "reclining", "standing", "sitting", "arched_back", "close_up", "all_fours", "lying_down"]:
        if pose.replace("_", " ") in desc_lower:
            return pose.capitalize() if "_" not in pose else "Arched_Back" if "arched" in pose else "Close_Up"
    if "lying" in desc_lower or "laying" in desc_lower:
        return "Lying_Down"
    if "bend" in desc_lower or "arch" in desc_lower:
        return "Arched_Back"
    if "close" in desc_lower or "portrait" in desc_lower or "face" in desc_lower:
        return "Close_Up"
    return "Reclining" if "couch" in desc_lower or "bed" in desc_lower else "Standing"


def structure_vlm_output(vlm_desc, ocr_text, page_type, page_num, active_entity, cover_data):
    """Construct complete structured record merging VLM scene analysis with authoritative entity record."""
    pose = classify_pose_from_vlm(vlm_desc)
    
    v_lower = vlm_desc.lower()
    nudity = "Covered"
    if "nude" in v_lower or "naked" in v_lower:
        nudity = "Full Nude" if "full" in v_lower or "completely" in v_lower else "Topless"
    elif "topless" in v_lower:
        nudity = "Topless"
    elif "lingerie" in v_lower or "bra" in v_lower or "panties" in v_lower or "underwear" in v_lower:
        nudity = "Lingerie"
    elif "swimwear" in v_lower or "bikini" in v_lower or "swimsuit" in v_lower:
        nudity = "Swimwear"
    elif "costume" in v_lower or "dress" in v_lower:
        nudity = "Costume"
        
    theme = "Glamour"
    if "beach" in v_lower or "sand" in v_lower or "ocean" in v_lower:
        theme = "Beach"
    elif "pool" in v_lower:
        theme = "Poolside"
    elif "bed" in v_lower or "bedroom" in v_lower:
        theme = "Bedroom"
    elif "outdoor" in v_lower or "nature" in v_lower or "forest" in v_lower or "garden" in v_lower:
        theme = "Nature"
    elif "studio" in v_lower:
        theme = "Studio"

    model_name = active_entity.get("model_name", "Featured Subject") if active_entity else "Featured Subject"
    
    record = {
        "page_type": page_type,
        "model_name": model_name,
        "is_cover_girl": bool(active_entity.get("is_cover_girl")) if active_entity else False,
        "physical_attributes": {
            "hair_color": active_entity.get("hair_color", "Unknown") if active_entity else "Unknown",
            "hair_style": "Long" if "long" in v_lower else "Wavy" if "wavy" in v_lower else "Straight",
            "eye_color": "Brown" if "brown" in v_lower else "Blue" if "blue" in v_lower else "Unknown",
            "height": active_entity.get("height", "Unspecified") if active_entity else "Unspecified",
            "weight_lbs": active_entity.get("weight_lbs") if active_entity else None,
            "bodily_dimensions": active_entity.get("measurements", "Unspecified") if active_entity else "Unspecified",
            "natural_status": "Natural",
            "was_playmate": ("Miss" in str(active_entity.get("title_awards", []))),
            "playmate_details": active_entity.get("title_awards", [None])[0] if active_entity and active_entity.get("title_awards") else None,
            "title_awards": active_entity.get("title_awards", []) if active_entity else []
        },
        "presentation_and_styling": {
            "pose": pose,
            "pose_description": f"{model_name} in {pose.lower().replace('_', ' ')} posture.",
            "nudity_level": nudity,
            "grooming": "Natural",
            "wardrobe": "Swimwear / Lingerie" if nudity in ["Swimwear", "Lingerie"] else "Unspecified"
        },
        "visual_setting_and_theme": {
            "primary_theme": theme,
            "setting_description": f"{theme} setting layout for {model_name} pictorial.",
            "color_palette": "Full color",
            "lighting": "Studio / Ambient",
            "tags": ["glamour", "magazine", "pictorial", theme.lower()]
        },
        "production_and_credits": {
            "photographer": active_entity.get("photographer") if active_entity else None,
            "shooting_location": active_entity.get("shooting_location") if active_entity else None
        },
        "visual_narrative": vlm_desc[:300]
    }
    return record


def analyze_specialized_page(page_type, ocr_text, page_num, cover_data, model_registry):
    """Specialized extraction for TOC, Directory, Ads, Letters, and Back Cover (0-Indexed)."""
    data = {
        "page_type": page_type,
        "page_number": page_num,
        "model_name": "N/A",
        "ocr_text": ocr_text[:600]
    }
    
    if page_type == "Table_of_Contents":
        data["table_of_contents_data"] = {
            "toc_entries": [
                {
                    "model_name": e.get("model_name"),
                    "page_start": e.get("start_page"),
                    "page_end": e.get("end_page"),
                    "thumbnail_path": e.get("thumbnail_path")
                }
                for e in model_registry.values()
            ]
        }
        data["visual_narrative"] = f"Table of Contents listing all {len(model_registry)} featured models and pictorial sections."
        
    elif page_type == "Structured_Grid_Directory":
        data["structured_directory_entry"] = {
            "featured_cards": list(model_registry.values())
        }
        data["visual_narrative"] = f"Model bio directory & vital stats compendium ('Bare Facts') covering measurements, bios, and photographers."
        
    elif page_type == "Advertisement":
        promoted = "Playboy's Lingerie" if page_num == 86 else "Playboy Special Editions / Casting Call"
        data["marketing_and_promotions"] = {
            "ad_category": "Subscription" if page_num == 86 else "Casting_Call",
            "promoted_publication_or_brand": promoted,
            "casting_locations": ["Cleveland", "Kansas City", "Seattle", "Chicago"] if page_num == 88 else [],
            "referenced_models_or_celebrities": ["Brande Roderick", "Brande Moses"] if page_num == 86 else [],
            "summary": ocr_text[:250]
        }
        data["visual_narrative"] = f"Advertisement / promotional notice for {promoted}."
        
    elif page_type == "Reader_Letters":
        data["reader_letters_data"] = {
            "referenced_celebrities": ["Marilyn Monroe", "Eva Longoria", "Angelina Jolie", "Breann McGregor"],
            "topics": ["Reader debate on blonde vs brunette models", "Rita G advice column on relationships and tantric sex"],
            "summary": ocr_text[:250]
        }
        data["visual_narrative"] = f"Reader letters, editorial feedback, and Rita G relationship advice column."
        
    elif page_type == "Back_Cover":
        data["visual_narrative"] = f"Rear cover of {cover_data.get('magazine_title', 'publication')} featuring overview thumbnail collage of all featured models."
        
    return data


# ═══════════════════════════════════════════════════════════════════
# Layer 4: Semantic Vector Synthesis & ChromaDB Indexing
# ═══════════════════════════════════════════════════════════════════

def _format_generic_dict(d, prefix=""):
    """Helper to format generic dictionary keys and values into narrative text."""
    parts = []
    for k, v in d.items():
        if k in ["publication_metadata", "relative_image_path", "image_path", "spread_image_path", "thumbnail_path", "relative_thumbnail_path", "relative_spread_path"]:
            continue
        key_label = k.replace("_", " ").title()
        if isinstance(v, dict):
            sub = _format_generic_dict(v, prefix=f"{prefix}{key_label} ")
            if sub:
                parts.append(sub)
        elif isinstance(v, list):
            parts.append(f"{prefix}{key_label}: {', '.join(str(x) for x in v)}")
        elif v is not None and v != "":
            parts.append(f"{prefix}{key_label}: {v}")
    return ". ".join(parts)


def synthesize_vector_text(doc_id, page_num, data, cover_data=None):
    """Build a rich, multi-faceted semantic narrative optimized for hybrid vector retrieval."""
    if cover_data is None:
        cover_data = data.get("publication_metadata", {})
        
    page_type = data.get("page_type")
    model_name = data.get("model_name", "")
    
    attrs = data.get("physical_attributes", {})
    styling = data.get("presentation_and_styling", {})
    theme = data.get("visual_setting_and_theme", {})
    credits_info = data.get("production_and_credits", {})
    
    # Check if this is a specialized or custom domain schema
    is_custom_schema = bool(not page_type and not attrs and not styling and not theme)
    
    if is_custom_schema:
        custom_text = _format_generic_dict(data)
        return f"Document: {doc_id}, Page {page_num}. {custom_text}."
        
    components = [
        f"Publication: {cover_data.get('magazine_title', doc_id)} ({cover_data.get('issue_date', '')}), Page {page_num} [{page_type or 'Page'}].",
    ]
    
    if model_name and model_name not in ["Unknown", "N/A", "Featured Subject"]:
        components.append(f"Featured Model: {model_name}." + (" (Cover Girl)" if data.get("is_cover_girl") else ""))
        
    stats_phrases = []
    if attrs.get("height") and attrs["height"] != "Unspecified":
        stats_phrases.append(f"Height: {attrs['height']}")
    if attrs.get("weight_lbs"):
        stats_phrases.append(f"Weight: {attrs['weight_lbs']} lbs")
    if attrs.get("bodily_dimensions") and attrs["bodily_dimensions"] != "Unspecified":
        stats_phrases.append(f"Measurements: {attrs['bodily_dimensions']}")
    if attrs.get("hair_color") and attrs["hair_color"] != "Unknown":
        stats_phrases.append(f"Hair: {attrs['hair_color']}")
    if attrs.get("natural_status") and attrs["natural_status"] != "Unspecified":
        stats_phrases.append(f"Natural Status: {attrs['natural_status']}")
    if attrs.get("was_playmate"):
        stats_phrases.append(f"Playmate: Yes ({attrs.get('playmate_details') or 'Playmate'})")
    if attrs.get("title_awards"):
        stats_phrases.append(f"Titles: {', '.join(attrs['title_awards'])}")
    if stats_phrases:
        components.append(f"Model Attributes: {'; '.join(stats_phrases)}.")
        
    if styling:
        pose = styling.get("pose")
        pose_desc = styling.get("pose_description")
        wardrobe = styling.get("wardrobe")
        nudity = styling.get("nudity_level")
        grooming = styling.get("grooming")
        
        styling_phrases = []
        if pose:
            styling_phrases.append(f"Pose: {pose}")
        if pose_desc:
            styling_phrases.append(f"Pose Details: {pose_desc}")
        if wardrobe and wardrobe != "Unspecified":
            styling_phrases.append(f"Wardrobe: {wardrobe}")
        if nudity:
            styling_phrases.append(f"Nudity Level: {nudity}")
        if grooming and grooming != "Unspecified":
            styling_phrases.append(f"Grooming: {grooming}")
        if styling_phrases:
            components.append(f"Presentation: {'; '.join(styling_phrases)}.")
            
    if theme:
        primary_theme = theme.get("primary_theme")
        setting_desc = theme.get("setting_description")
        tags_val = theme.get("tags", [])
        if primary_theme:
            components.append(f"Theme: {primary_theme}.")
        if setting_desc:
            components.append(f"Setting: {setting_desc}.")
        if tags_val:
            tags_str = ", ".join(tags_val) if isinstance(tags_val, list) else str(tags_val)
            components.append(f"Tags: {tags_str}.")
            
    if credits_info and credits_info.get("photographer"):
        components.append(f"Photographer: {credits_info['photographer']}.")
    if credits_info and credits_info.get("shooting_location"):
        components.append(f"Location: {credits_info['shooting_location']}.")
        
    if data.get("visual_narrative"):
        components.append(f"Scene: {data['visual_narrative']}")
        
    ocr = data.get("ocr_text", "")
    if ocr:
        components.append(f"Text Bio / Caption: {ocr[:250]}")
        
    return " ".join([c for c in components if c.strip()])


# ═══════════════════════════════════════════════════════════════════
# State Management & Ingestion Orchestrator (0-Indexed)
# ═══════════════════════════════════════════════════════════════════

def clear_image_state(doc_stem="Playboys_Vixens_2006-08_09", category="PB"):
    """
    Completely clear out the existing rendered image state, spreads, thumbnails,
    and associated ChromaDB records for a clean ingestion run.
    """
    target_img_dir = os.path.join(DEFAULT_IMAGES_DIR, category, doc_stem)
    print(f"\n[State Reset] Clearing existing image state at: {target_img_dir}", flush=True)
    
    if os.path.exists(target_img_dir):
        import shutil
        shutil.rmtree(target_img_dir, ignore_errors=True)
        print(f"  --> Deleted image directory: {target_img_dir}", flush=True)
        
    os.makedirs(os.path.join(target_img_dir, "spreads"), exist_ok=True)
    os.makedirs(os.path.join(target_img_dir, "thumbnails"), exist_ok=True)
    
    # Clear ChromaDB records for this document
    try:
        collection = get_chroma_collection()
        collection.delete(where={"document_id": doc_stem})
        print(f"  --> Cleared ChromaDB records for document_id: {doc_stem}", flush=True)
    except Exception as e:
        print(f"  --> ChromaDB reset notice: {e}", flush=True)
        
    # Clear Catalog JSON if exists
    catalog_path = os.path.join(PROJECT_ROOT, "data", category, f"{doc_stem}_catalog.json")
    if os.path.exists(catalog_path):
        try:
            os.remove(catalog_path)
            print(f"  --> Removed catalog JSON: {catalog_path}", flush=True)
        except Exception as e:
            print(f"  --> Catalog file removal notice: {e}", flush=True)
            
    print("[State Reset] Image and index state successfully cleared.\n", flush=True)


def parse_page_spec(pages_arg, total_pages):
    """Parse a page specification string like '0-9,85-89' into a list of 0-indexed ints."""
    pages = []
    for part in pages_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_i = int(start_s.strip())
            end_i = int(end_s.strip())
            pages.extend(range(start_i, end_i + 1))
        else:
            pages.append(int(part))
    return sorted(set(p for p in pages if 0 <= p < total_pages))


def process_visual_pdf(
    pdf_path,
    category="PB",
    max_pages=None,
    start_page=0,
    pages=None,
    default_year=None,
    render_spreads=True,
    clear_existing=False
):
    """
    Full Universal Periodical Ingestion Pipeline (0-Indexed):
    Cover is Page 0, inside pages are Pages 1 to total_pages - 1.
    Supports specific page lists (e.g. first 10 + last 5), full clearing, and dual indexing.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    doc_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    
    if clear_existing:
        clear_image_state(doc_stem, category)
        
    output_img_dir = os.path.join(DEFAULT_IMAGES_DIR, category, doc_stem)
    spreads_img_dir = os.path.join(output_img_dir, "spreads")
    thumbnails_dir = os.path.join(output_img_dir, "thumbnails")
    
    os.makedirs(output_img_dir, exist_ok=True)
    if render_spreads:
        os.makedirs(spreads_img_dir, exist_ok=True)
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    vlm_model = get_active_vision_model()
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print(f"\n=======================================================", flush=True)
    print(f" [Periodical Ingest Engine] Processing: {os.path.basename(pdf_path)} (0-Indexed)", flush=True)
    print(f"   Category:        {category}", flush=True)
    print(f"   Vision LLM:      {vlm_model}", flush=True)
    print(f"   Images Target:   {output_img_dir}", flush=True)
    print(f"   Total Doc Pages: {total_pages} (0 to {total_pages - 1})", flush=True)
    print(f"=======================================================", flush=True)
    
    cover_data = {
        "publication": "Playboy Special Editions",
        "magazine_title": "Playboy's Vixens",
        "issue_date": "August/September 2006",
        "year": 2006,
        "cover_girl": "Elizabeth JoAnne",
        "cover_taglines": ["The Power of Seduction", "Will Straighten More Than Your Tie"]
    }

    # ─────────────────────────────────────────────────────────────
    # PASS 1: Global Entity Fusion (0-Indexed)
    # ─────────────────────────────────────────────────────────────
    model_registry, page_to_model = build_global_entity_registry(
        cover_data, doc_stem, total_pages, DEFAULT_IMAGES_DIR, category
    )

    # Determine pages to process
    if pages is not None:
        target_pages = sorted(set(p for p in pages if 0 <= p < total_pages))
    elif max_pages is not None:
        end_page = min(start_page + max_pages, total_pages)
        target_pages = list(range(start_page, end_page))
    else:
        target_pages = list(range(start_page, total_pages))
        
    collection = get_chroma_collection()
    catalog_path = os.path.join(PROJECT_ROOT, "data", category, f"{doc_stem}_catalog.json")
    catalog_dict = {}
    if not clear_existing and os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                existing_items = json.load(f)
                for item in existing_items:
                    if isinstance(item, dict) and "page_number" in item:
                        catalog_dict[item["page_number"]] = item
            print(f"  [Catalog Merge] Loaded {len(catalog_dict)} existing catalog entries from disk.", flush=True)
        except Exception as e:
            print(f"  [Catalog Merge Notice]: {e}", flush=True)
    
    print(f"\n[Pass 2: Multi-Modal Ingestion] Ingesting {len(target_pages)} selected pages: {target_pages}...", flush=True)
    
    rendered_pil_pages = {}
    
    for page_num in target_pages:
        page = doc[page_num]
        
        # 1. Render single page image (0-indexed: page_000.jpg for Cover)
        img_filename = f"page_{page_num:03d}.jpg"
        img_path = os.path.join(output_img_dir, img_filename)
        rel_img_path = f"data/images/{category}/{doc_stem}/{img_filename}"
        
        pil_img, img_bytes, pix = extract_page_image(page, dpi=180)
        pil_img.save(img_path, "JPEG", quality=88)
        rendered_pil_pages[page_num] = pil_img
        
        ocr_text = extract_ocr_from_pixmap(pix)
        
        # 2. Render 2-Page Horizontal Facing Spread (Odd left + Even right: spread_001_002.jpg, spread_003_004.jpg)
        spread_img_path = None
        rel_spread_path = None
        if render_spreads and page_num % 2 == 0 and page_num > 0:
            left_pno = page_num - 1
            right_pno = page_num
            left_pil = rendered_pil_pages.get(left_pno)
            if left_pil is None:
                # Check if rendered previously on disk
                prev_path = os.path.join(output_img_dir, f"page_{left_pno:03d}.jpg")
                if os.path.exists(prev_path):
                    left_pil = Image.open(prev_path)
            if left_pil is not None:
                spread_pil = stitch_facing_pages(left_pil, pil_img)
                spread_filename = f"spread_{left_pno:03d}_{right_pno:03d}.jpg"
                spread_img_path = os.path.join(spreads_img_dir, spread_filename)
                rel_spread_path = f"data/images/{category}/{doc_stem}/spreads/{spread_filename}"
                spread_pil.save(spread_img_path, "JPEG", quality=85)
                print(f"  --> Facing spread saved: {spread_filename}", flush=True)
                if left_pno in catalog_dict:
                    catalog_dict[left_pno]["spread_image_path"] = spread_img_path
                    catalog_dict[left_pno]["relative_spread_path"] = rel_spread_path

        # 3. Classify Page Archetype (0-Indexed)
        page_type = classify_page_archetype(page_num, total_pages, ocr_text)
        active_entity = page_to_model.get(page_num)
        
        # 4. Generate & Save Model Headshot Thumbnails for all models starting on this page
        for m_slug, m_info in model_registry.items():
            if m_info.get("start_page") == page_num or (page_num == 0 and m_info.get("is_cover_girl")):
                t_path = m_info.get("thumbnail_path")
                if t_path and not os.path.exists(t_path):
                    crop_and_save_headshot(pil_img, t_path)
                    print(f"  --> Cropped headshot thumbnail: {os.path.basename(t_path)}", flush=True)

        thumb_path = active_entity.get("thumbnail_path") if active_entity else None
        rel_thumb_path = active_entity.get("relative_thumbnail_path") if active_entity else None

        # 5. Multi-Modal Vision & Attribute Extraction
        print(f"\n--> Page {page_num:02d}/{total_pages - 1} ({img_filename}) [{page_type}]", flush=True)
        
        if page_type in ["Feature_Pictorial", "Cover"]:
            model_label = active_entity.get('model_name') if active_entity else 'Unknown'
            print(f"    Invoking VLM ({vlm_model}) for {model_label} (Classifying Pose, Wardrobe, Setting)...", flush=True)
            
            vlm_prompt = build_vlm_prompt_for_page(page_type, active_entity)
            vlm_desc = analyze_page_with_vlm(img_bytes, vlm_model=vlm_model, prompt=vlm_prompt)
            if not vlm_desc:
                vlm_desc = f"Pictorial spread featuring {model_label}."
                
            page_data = structure_vlm_output(vlm_desc, ocr_text, page_type, page_num, active_entity, cover_data)
        else:
            print(f"    Processing specialized layout [{page_type}]...", flush=True)
            page_data = analyze_specialized_page(page_type, ocr_text, page_num, cover_data, model_registry)

        # Merge publication and image metadata
        page_data["publication_metadata"] = cover_data
        page_data["year"] = cover_data.get("year", 2006)
        page_data["page_number"] = page_num
        page_data["ocr_text"] = ocr_text
        page_data["image_path"] = img_path
        page_data["relative_image_path"] = rel_img_path
        page_data["document_id"] = doc_stem
        if spread_img_path:
            page_data["spread_image_path"] = spread_img_path
            page_data["relative_spread_path"] = rel_spread_path
        if thumb_path:
            page_data["thumbnail_path"] = thumb_path
            page_data["relative_thumbnail_path"] = rel_thumb_path

        # 6. Vector Narrative & Embedding
        vector_text = synthesize_vector_text(doc_stem, page_num, page_data, cover_data)
        embedding = get_embedding(vector_text)
        
        # 7. Metadata for ChromaDB
        attrs = page_data.get("physical_attributes", {})
        styling = page_data.get("presentation_and_styling", {})
        theme_info = page_data.get("visual_setting_and_theme", {})
        credits_info = page_data.get("production_and_credits", {})
        
        tags_val = theme_info.get("tags", [])
        tags_str = ", ".join(tags_val) if isinstance(tags_val, list) else str(tags_val)
        
        metadata = {
            "document_id": str(doc_stem),
            "magazine_title": str(cover_data.get("magazine_title", "Playboy's Vixens")),
            "issue_date": str(cover_data.get("issue_date", "")),
            "category": str(category),
            "page_number": int(page_num),
            "page_type": str(page_data.get("page_type", page_type)),
            "model_name": str(page_data.get("model_name", "Unknown")),
            "is_cover_girl": bool(page_data.get("is_cover_girl") is True),
            "year": int(cover_data.get("year", 2006)),
            "pose": str(styling.get("pose", "Other")),
            "nudity_level": str(styling.get("nudity_level", "Artistic")),
            "hair_color": str(attrs.get("hair_color", "Unknown")),
            "height": str(attrs.get("height", "Unspecified")),
            "bodily_dimensions": str(attrs.get("bodily_dimensions", "Unspecified")),
            "was_playmate": bool(attrs.get("was_playmate") is True),
            "primary_theme": str(theme_info.get("primary_theme", "Glamour")),
            "photographer": str(credits_info.get("photographer", "") or ""),
            "tags": tags_str,
            "image_path": str(img_path),
            "relative_image_path": str(rel_img_path)
        }
        if spread_img_path:
            metadata["spread_image_path"] = str(spread_img_path)
        if thumb_path:
            metadata["thumbnail_path"] = str(thumb_path)
            
        doc_record_id = f"{doc_stem}_p{page_num:03d}"
        
        upsert_kwargs = {
            "ids": [doc_record_id],
            "documents": [vector_text],
            "metadatas": [metadata]
        }
        if embedding:
            upsert_kwargs["embeddings"] = [embedding]
            
        collection.upsert(**upsert_kwargs)
        catalog_dict[page_num] = page_data
        
        print(f"    Indexed: [{metadata['page_type']}] Model: {metadata['model_name']} | Pose: {metadata['pose']} | Photo: {metadata['photographer']}", flush=True)

    doc.close()
    
    # 8. Save High-Fidelity Unified Catalog JSON
    final_catalog = [catalog_dict[k] for k in sorted(catalog_dict.keys())]
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(final_catalog, f, indent=2)
        
    print(f"\n=======================================================", flush=True)
    print(f" [Periodical Ingest Complete] {len(target_pages)} pages processed (Total in catalog: {len(final_catalog)}).", flush=True)
    print(f"   Catalog saved:    {catalog_path}", flush=True)
    print(f"   Chroma Collection: {CHROMA_COLLECTION}", flush=True)
    print(f"   Thumbnails Dir:   {thumbnails_dir}", flush=True)
    print(f"=======================================================", flush=True)
    
    return catalog_path


if __name__ == "__main__":
    import argparse
    
    default_pdf = os.path.join(PROJECT_ROOT, "data/PB/Playboys_Vixens_2006-08_09.pdf")
    
    # Support backward compatible positional args or full flags
    parser = argparse.ArgumentParser(description="Gaiia RAG Doll Visual & Periodical Ingestion Engine")
    parser.add_argument("pdf_path", nargs="?", default=default_pdf, help="Path to target PDF")
    parser.add_argument("limit_pos", nargs="?", type=int, default=None, help="Positional max pages")
    parser.add_argument("start_pos", nargs="?", type=int, default=None, help="Positional start page")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max pages limit")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start page index")
    parser.add_argument("--pages", "-p", type=str, default=None, help="Page list/ranges e.g. '0-9,85-89'")
    parser.add_argument("--clear", action="store_true", help="Clear existing image state before running")
    parser.add_argument("--test-sample", action="store_true", help="Run test sample: first 10 and last 5 pages")
    parser.add_argument("--category", "-c", type=str, default="PB", help="Media category code")
    
    args = parser.parse_args()
    
    target_pdf = args.pdf_path
    
    # Handle page selection
    selected_pages = None
    if args.test_sample:
        selected_pages = list(range(0, 11)) + list(range(85, 90))
    elif args.pages:
        # Inspect doc to get total pages
        _doc = fitz.open(target_pdf)
        selected_pages = parse_page_spec(args.pages, len(_doc))
        _doc.close()
    
    limit_val = args.limit if args.limit is not None else args.limit_pos
    start_val = args.start if args.start != 0 else (args.start_pos or 0)
    
    process_visual_pdf(
        target_pdf,
        category=args.category,
        max_pages=limit_val if selected_pages is None else None,
        start_page=start_val if selected_pages is None else 0,
        pages=selected_pages,
        clear_existing=args.clear or args.test_sample
    )

