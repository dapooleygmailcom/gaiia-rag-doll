import os
import fitz  # PyMuPDF
import chromadb
import ollama

POLICIES_DIR = "data/policies"
CHROMA_DB_DIR = "data/chroma"
CHROMA_COLLECTION_NAME = "policy-comparison"

FILENAME_TO_CARRIER = {
    "aami-home-contents-insurance-pds.pdf": "AAMI",
    "cba-home-pds.pdf": "CBA",
    "FSR_HomeContentInsPDS.pdf": "Westpac_FSR",
    "FSR_HomeContentInsPDS_pre21Dec2025.pdf": "Westpac_FSR_Pre2025",
    "home-and-contents-insurance-pds.pdf": "Allianz_NAB",
    "home-insurance-ped-current.pdf": "RACV_PED",
    "pds.pdf": "CGU_ANZ",
    "POL1388TIO.pdf": "TIO"
}

def setup_vector_db():
    print("Initializing ChromaDB at:", CHROMA_DB_DIR)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Clear existing collection if it exists to start fresh
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print("Cleared existing collection to prepare for section-based rebuild.")
    except Exception:
        pass
        
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def get_body_size(doc):
    """
    Find the most common font size (the mode) in the PDF by sampling character count.
    """
    sizes = {}
    pages_to_sample = min(15, len(doc))
    for page_idx in range(pages_to_sample):
        page = doc[page_idx]
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        size = round(s["size"], 1)
                        sizes[size] = sizes.get(size, 0) + len(s["text"])
    if not sizes:
        return 10.0  # Fallback
    body_size = max(sizes, key=sizes.get)
    return body_size

def extract_section_chunks(doc, carrier, f_name):
    """
    Parse PDF dynamically by Section and Subsection using font sizes.
    """
    body_size = get_body_size(doc)
    print(f"  Detected body font size: {body_size}pt")
    
    chunks = []
    current_section = "Introduction"
    current_subsection = ""
    accumulated_lines = []
    accumulated_chars = 0
    start_page = 1
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        blocks = page.get_text("dict").get("blocks", [])
        
        # Sort blocks top-to-bottom
        blocks = sorted(blocks, key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
        
        for b in blocks:
            if "lines" not in b:
                continue
            
            # Sort lines in block top-to-bottom
            lines = sorted(b["lines"], key=lambda l: l.get("bbox", [0, 0, 0, 0])[1])
            
            for l in lines:
                # Group text and find max font size in the line
                line_text = ""
                max_size = 0
                for s in l["spans"]:
                    line_text += s["text"]
                    if s["size"] > max_size:
                        max_size = s["size"]
                
                line_text = line_text.strip()
                if not line_text:
                    continue
                    
                # Skip header/footer noise (very small text)
                if max_size < 7.0:
                    continue
                    
                # Detect heading
                is_heading = False
                heading_level = 0  # 1 = Section, 2 = Subsection
                
                # Heuristic: line is a heading if it is larger than body text
                # and is not excessively long
                if max_size >= body_size * 1.15 and len(line_text) < 120:
                    is_heading = True
                    if max_size >= body_size * 1.4:
                        heading_level = 1
                    else:
                        heading_level = 2
                
                # Check for standard uppercase section prefixes or typical patterns
                if not is_heading:
                    text_upper = line_text.upper()
                    if (text_upper.startswith("SECTION ") or text_upper.startswith("PART ")) and len(line_text) < 40:
                        is_heading = True
                        heading_level = 1
                        
                if is_heading:
                    # Save current chunk if we have enough content
                    if accumulated_chars >= 150:
                        text_content = "\n".join(accumulated_lines)
                        chunks.append({
                            "text": f"[Carrier: {carrier}] [Section: {current_section}] [Subsection: {current_subsection}] {text_content}",
                            "metadata": {
                                "carrier": carrier,
                                "source": f_name,
                                "section": current_section,
                                "subsection": current_subsection,
                                "page": start_page
                            }
                        })
                        accumulated_lines = []
                        accumulated_chars = 0
                        
                    # Update active heading
                    if heading_level == 1:
                        current_section = line_text
                        current_subsection = ""
                        start_page = page_num
                    else:
                        current_subsection = line_text
                        start_page = page_num
                else:
                    # Accumulate regular text
                    accumulated_lines.append(line_text)
                    accumulated_chars += len(line_text)
                    
                    # Split if chunk is too large (to maintain semantic resolution)
                    if accumulated_chars >= 2500:
                        text_content = "\n".join(accumulated_lines)
                        chunks.append({
                            "text": f"[Carrier: {carrier}] [Section: {current_section}] [Subsection: {current_subsection}] {text_content}",
                            "metadata": {
                                "carrier": carrier,
                                "source": f_name,
                                "section": current_section,
                                "subsection": current_subsection,
                                "page": start_page
                            }
                        })
                        # Overlap: keep the last 2 lines
                        overlap_lines = accumulated_lines[-2:] if len(accumulated_lines) >= 2 else accumulated_lines[-1:]
                        accumulated_lines = list(overlap_lines)
                        accumulated_chars = sum(len(x) for x in accumulated_lines)
                        start_page = page_num
                        
    # Append final chunk
    if accumulated_lines:
        text_content = "\n".join(accumulated_lines)
        chunks.append({
            "text": f"[Carrier: {carrier}] [Section: {current_section}] [Subsection: {current_subsection}] {text_content}",
            "metadata": {
                "carrier": carrier,
                "source": f_name,
                "section": current_section,
                "subsection": current_subsection,
                "page": start_page
            }
        })
        
    return chunks

def ingest_policies():
    collection = setup_vector_db()
    
    if not os.path.exists(POLICIES_DIR):
        print(f"Error: Directory {POLICIES_DIR} does not exist.")
        return
        
    pdf_files = [f for f in os.listdir(POLICIES_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files in policies directory.")
    
    total_chunks_indexed = 0
    
    for idx, f_name in enumerate(pdf_files, 1):
        f_path = os.path.join(POLICIES_DIR, f_name)
        carrier = FILENAME_TO_CARRIER.get(f_name, f_name.replace(".pdf", "").upper())
        
        print(f"\n[{idx}/{len(pdf_files)}] Parsing '{f_name}' for Carrier: {carrier}...")
        
        try:
            doc = fitz.open(f_path)
            chunks = extract_section_chunks(doc, carrier, f_name)
            doc.close()
            
            print(f"  Extracted {len(chunks)} hierarchical section chunks.")
            
            batch_ids = []
            batch_embeddings = []
            batch_metadatas = []
            batch_documents = []
            
            for chunk_idx, chunk in enumerate(chunks, 1):
                text = chunk["text"]
                metadata = chunk["metadata"]
                
                # Get embeddings from Ollama
                try:
                    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
                    embedding = response["embedding"]
                except Exception as e:
                    print(f"    Error generating embedding on chunk {chunk_idx}: {e}")
                    continue
                
                doc_id = f"{carrier.lower()}_chunk_{chunk_idx}"
                
                batch_ids.append(doc_id)
                batch_embeddings.append(embedding)
                batch_metadatas.append(metadata)
                batch_documents.append(text)
                
                # Upsert in batches of 20
                if len(batch_ids) >= 20:
                    collection.upsert(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        documents=batch_documents
                    )
                    batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []
            
            # Upsert remaining
            if batch_ids:
                collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
                
            total_chunks_indexed += len(chunks)
            print(f"  Carrier {carrier} ingestion complete.")
            
        except Exception as e:
            print(f"Error processing document {f_name}: {str(e)}")
            
    print(f"\nIngestion of all policies complete! Total chunks indexed: {total_chunks_indexed}")

if __name__ == "__main__":
    ingest_policies()
