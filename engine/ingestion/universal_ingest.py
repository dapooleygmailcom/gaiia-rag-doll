"""
Universal Autonomous Ingestion Pipeline — Gaiia RAG Doll.

Auto-detects document characteristics (visual imagery, dense text, rules,
tables/numbers, legal policies) and automatically executes the optimal
ingestion and indexing strategy without requiring manual user flags.
"""

import os
import re
import sys
import json
import fitz  # PyMuPDF

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))

# Note: Specialized pipelines (e.g. ingest_visual, ingest_rules) are imported lazily inside route handlers




class DocumentProfiler:
    """Inspects raw documents and determines structural composition."""
    
    @staticmethod
    def profile_pdf(pdf_path, sample_limit=8):
        """
        Analyze a PDF file and return diagnostic metrics and the classified pipeline type.
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        pages_to_sample = sorted(set([
            0,
            1,
            min(2, total_pages - 1),
            total_pages // 4,
            total_pages // 2,
            min(total_pages - 1, 3 * total_pages // 4)
        ]))
        pages_to_sample = [p for p in pages_to_sample if 0 <= p < total_pages][:sample_limit]
        
        total_chars = 0
        total_images = 0
        total_image_coverage = 0.0
        total_numbers = 0
        rule_pattern_hits = 0
        policy_keyword_hits = 0
        
        rule_regex = re.compile(r'(?:^|\s)(?:[A-Z]\d{1,2}\.\d{1,4}|\d{1,2}\.\d{1,3}(?:\.\d{1,2})?)\b')
        policy_keywords = ["product disclosure", "pds", "policy", "coverage", "exclusion", "sum insured", "definitions", "clause", "claim"]
        
        for pno in pages_to_sample:
            page = doc[pno]
            page_rect = page.rect
            page_area = page_rect.width * page_rect.height if page_rect.width * page_rect.height > 0 else 1.0
            
            # Text analysis
            text = page.get_text()
            chars = len(text.strip())
            total_chars += chars
            
            # Number analysis
            digits = len(re.findall(r'\d+', text))
            total_numbers += digits
            
            # Rule pattern matches
            rule_pattern_hits += len(rule_regex.findall(text))
            
            # Policy keyword matches
            text_lower = text.lower()
            for kw in policy_keywords:
                if kw in text_lower:
                    policy_keyword_hits += 1
                    
            # Image and visual layout analysis
            images = page.get_images(full=True)
            total_images += len(images)
            
            # Approximate visual surface area from image bboxes or drawing blocks
            blocks = page.get_text("dict").get("blocks", [])
            img_area = 0.0
            for b in blocks:
                if b.get("type") == 1:  # image block
                    bbox = b.get("bbox", [0, 0, 0, 0])
                    w = max(0, bbox[2] - bbox[0])
                    h = max(0, bbox[3] - bbox[1])
                    img_area += (w * h)
                    
            coverage = min(1.0, img_area / page_area)
            # If PyMuPDF found image objects but dict block was empty, estimate coverage
            if coverage == 0 and len(images) > 0:
                coverage = min(1.0, 0.45 * len(images))
                
            total_image_coverage += coverage
            
        doc.close()
        
        sample_count = max(1, len(pages_to_sample))
        avg_chars_per_page = total_chars / sample_count
        avg_images_per_page = total_images / sample_count
        avg_coverage = total_image_coverage / sample_count
        numeric_ratio = total_numbers / max(1, total_chars)
        
        # Classification Decision Tree
        pipeline_type = "UNKNOWN"
        confidence_reason = ""
        
        filename_lower = os.path.basename(pdf_path).lower()

        is_visual_media = any(term in filename_lower for term in [
            "vixen", "playboy", "glamour", "portfolio", "lookbook",
            "photo", "magazine", "comic", "catalog", "catalogue"
        ])

        # 1. Visual Media (Magazines, Art, Comics, Portfolios, Lookbooks)
        if is_visual_media and (avg_coverage >= 0.35 or avg_images_per_page >= 0.8 or avg_chars_per_page < 1200):
            pipeline_type = "VISUAL_MEDIA"
            confidence_reason = (
                f"Visual media publication ({avg_coverage:.1%} coverage, "
                f"{avg_images_per_page:.1f} images/pg) with visual-media filename signal"
            )
        elif is_visual_media:
            pipeline_type = "VISUAL_MEDIA"
            confidence_reason = "Visual-media keyword match in filename"

        # 2. Scanned Text — A document with <50 native chars/page AND high image coverage
        #    that is NOT an explicit visual media publication is a scanned document.
        elif avg_chars_per_page < 50 and avg_coverage > 0.5:
            pipeline_type = "SCANNED_OCR"
            confidence_reason = (
                f"Sparse native text ({avg_chars_per_page:.0f} chars/pg) with high image "
                f"coverage ({avg_coverage:.1%}) — almost certainly a scanned document requiring OCR"
            )

        # 4. Rulebook / Technical Specification
        elif rule_pattern_hits >= 3 or any(term in filename_lower for term in [
            "rule", "errata", "asl", "upfront", "codex", "manual"
        ]):
            pipeline_type = "RULEBOOK_TECHNICAL"
            confidence_reason = (
                f"Detected numbering schema ({rule_pattern_hits} pattern hits) "
                f"or rulebook filename indicators"
            )

        # 5. Policy / Legal / Insurance PDS
        elif policy_keyword_hits >= 3 or any(term in filename_lower for term in [
            "pds", "policy", "insurance", "disclosure", "contract", "legal"
        ]):
            pipeline_type = "POLICY_HIERARCHICAL"
            confidence_reason = f"Detected policy/legal ontology ({policy_keyword_hits} keyword hits)"

        # 6. Scanned Text fallback (low text, not already caught by guard above)
        elif avg_chars_per_page < 80:
            pipeline_type = "SCANNED_OCR"
            confidence_reason = f"Sparse native text ({avg_chars_per_page:.0f} chars/pg), likely requires OCR"

        # 7. Default General Text
        else:
            pipeline_type = "GENERAL_TEXT"
            confidence_reason = f"Standard text publication ({avg_chars_per_page:.0f} chars/pg)"

        profile = {
            "file_path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "total_pages": total_pages,
            "sampled_pages": len(pages_to_sample),
            "avg_chars_per_page": round(avg_chars_per_page, 1),
            "avg_images_per_page": round(avg_images_per_page, 2),
            "avg_visual_coverage": round(avg_coverage, 3),
            "numeric_ratio": round(numeric_ratio, 3),
            "rule_pattern_hits": rule_pattern_hits,
            "policy_keyword_hits": policy_keyword_hits,
            "detected_pipeline": pipeline_type,
            "reason": confidence_reason
        }
        
        return profile


class UniversalIngestionEngine:
    """Universal dispatcher that profiles and routes files to correct pipelines."""
    
    def __init__(self):
        self.profiler = DocumentProfiler()
        
    def ingest_file(self, file_path, category=None, max_pages=None, target=None, title_id=None, tenant_id=None, pre_extracted_text=None, pre_extracted_text_path=None):
        """Profile and automatically ingest a single file. Supports pre-extracted OCR text."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        print("\n" + "=" * 70)
        print(f" [Universal Ingest Engine] Inspecting: {os.path.basename(file_path)} (Target: {target or 'DEFAULT'})")
        print("=" * 70)
        
        if ext == ".pdf":
            profile = self.profiler.profile_pdf(file_path)
            print(f"  ├─ Total Pages:     {profile['total_pages']}")
            print(f"  ├─ Avg Text/Page:   {profile['avg_chars_per_page']} chars")
            print(f"  ├─ Visual Coverage: {profile['avg_visual_coverage']:.1%}")
            print(f"  ├─ Detected Route:  [{profile['detected_pipeline']}]")
            print(f"  └─ Reasoning:       {profile['reason']}")
            print("-" * 70)
            
            # Autonomous Routing
            route = profile["detected_pipeline"]
            
            if route == "VISUAL_MEDIA":
                cat = category or ("PB" if "pb" in file_path.lower() or "vixen" in file_path.lower() else "VISUAL")
                print(f"[Router] Activating VISUAL MEDIA PIPELINE (Category: {cat})...")
                try:
                    from engine.ingestion.ingest_visual import process_visual_pdf
                except ImportError:
                    from ingest_visual import process_visual_pdf
                return process_visual_pdf(file_path, category=cat, max_pages=max_pages)
                
            elif route == "RULEBOOK_TECHNICAL":
                print(f"[Router] Activating RULEBOOK & TECHNICAL SPEC PIPELINE (Target: {target or 'DEFAULT'})...")
                try:
                    from engine.ingestion.ingest_rules import process_rules_pdf
                    return process_rules_pdf(
                        file_path,
                        target=target,
                        game_id=title_id,
                        tenant_id=tenant_id,
                        pre_extracted_text=pre_extracted_text,
                        pre_extracted_text_path=pre_extracted_text_path
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    raise e
                    
            elif route == "POLICY_HIERARCHICAL":
                print("[Router] Activating HIERARCHICAL POLICY & LEGAL PIPELINE...")
                return profile
                
            elif route == "SCANNED_OCR":
                print(f"[Router] Activating SCANNED & OCR PIPELINE for {file_path} (Target: {target or 'DEFAULT'})...")
                try:
                    from engine.ingestion.ingest_rules import process_rules_pdf
                    return process_rules_pdf(
                        file_path,
                        target=target,
                        game_id=title_id,
                        tenant_id=tenant_id,
                        ocr_required=(not pre_extracted_text and not pre_extracted_text_path),
                        pre_extracted_text=pre_extracted_text,
                        pre_extracted_text_path=pre_extracted_text_path
                    )
                except Exception as e:
                    print(f"  SCANNED_OCR ingestion error: {e}")
                    import traceback
                    traceback.print_exc()
                    return profile
            else:
                print(f"[Router] Activating GENERAL TEXT PIPELINE for {file_path} (Target: {target or 'DEFAULT'})...")
                try:
                    from engine.ingestion.ingest_rules import process_rules_pdf
                    return process_rules_pdf(
                        file_path,
                        target=target,
                        game_id=title_id,
                        tenant_id=tenant_id,
                        pre_extracted_text=pre_extracted_text,
                        pre_extracted_text_path=pre_extracted_text_path
                    )
                except Exception as e:
                    print(f"  General text routing fallback: {e}")
                    return profile
                
        elif ext in [".csv", ".tsv", ".xlsx"]:
            print("[Router] Activating NUMERICAL & TABULAR PIPELINE...")
            try:
                from engine.ingestion.ingest_analysis import ingest_data
                return ingest_data()
            except Exception as e:
                print(f"  Tabular ingest error: {e}")
                return None
        else:
            print(f"[Router] Unsupported or generic file format: {ext}")
            return None

    def ingest_directory(self, dir_path, max_pages_per_doc=None):
        """Process all supported documents in a directory."""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
            
        supported_exts = [".pdf", ".csv", ".tsv", ".png", ".jpg"]
        files = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in supported_exts
        ]
        
        print(f"\n[Universal Ingest] Found {len(files)} candidate files in {dir_path}")
        results = []
        for f in files:
            res = self.ingest_file(f, max_pages=max_pages_per_doc)
            results.append((f, res))
            
        return results


if __name__ == "__main__":
    engine = UniversalIngestionEngine()
    
    if len(sys.argv) > 1:
        target = sys.argv[1]
        pages_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        
        if os.path.isdir(target):
            engine.ingest_directory(target, max_pages_per_doc=pages_limit)
        else:
            engine.ingest_file(target, max_pages=pages_limit)
    else:
        # Default test: run on data/PB/
        default_target = os.path.join(PROJECT_ROOT, "data/PB/Playboys_Vixens_2006-08_09.pdf")
        if os.path.exists(default_target):
            engine.ingest_file(default_target, max_pages=5)
        else:
            print("Usage: python universal_ingest.py <file_or_directory_path> [max_pages]")
