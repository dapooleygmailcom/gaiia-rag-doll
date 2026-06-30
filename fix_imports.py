import os
import re

directories = ['engine', 'tests', 'scripts']

replacements = {
    r'\bfrom rules_lawyer\b': 'from engine.retrieval.rules_lawyer',
    r'\bimport rules_lawyer\b': 'import engine.retrieval.rules_lawyer',
    r'\bfrom auto_discover\b': 'from engine.ingestion.auto_discover',
    r'\bimport auto_discover\b': 'import engine.ingestion.auto_discover',
    r'\bfrom ingest_rules\b': 'from engine.ingestion.ingest_rules',
    r'\bimport ingest_rules\b': 'import engine.ingestion.ingest_rules',
    r'\bfrom ocr_processor\b': 'from engine.ingestion.ocr_processor',
    r'\bimport ocr_processor\b': 'import engine.ingestion.ocr_processor',
    r'\bfrom policy_agent\b': 'from engine.retrieval.policy_agent',
    r'\bimport policy_agent\b': 'import engine.retrieval.policy_agent',
    r'\bfrom analysis_agent\b': 'from engine.retrieval.analysis_agent',
    r'\bimport analysis_agent\b': 'import engine.retrieval.analysis_agent'
}

for d in directories:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                original_content = content
                for pattern, replacement in replacements.items():
                    content = re.sub(pattern, replacement, content)
                
                if content != original_content:
                    with open(path, 'w', encoding='utf-8') as file:
                        file.write(content)
                    print(f'Updated {path}')
