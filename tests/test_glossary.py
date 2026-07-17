import fitz, ollama

def run_test():
    try:
        doc = fitz.open('data/asl/pdfcoffee.com_asl-2nd-edition-core-rules-pdf-free.pdf')
        text = doc[4].get_text() + '\n' + doc[5].get_text() + '\n' + doc[6].get_text()
        doc.close()
        
        prompt = f"""Extract any game-specific acronyms/abbreviations and their full meanings from this text.
        Return ONLY a JSON dictionary where keys are acronyms and values are full meanings.
        
        Text:
        {text[:8000]}
        """
        res = ollama.generate(model='llama3.1:8b', prompt=prompt)
        print(res['response'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_test()
