import os
import json
from app.config.data_config import PROJECT_ROOT

arabic_path = os.path.join(PROJECT_ROOT, "data", "raw-files", "raw-tags-arabic-indexed", "tags.json")
french_path = os.path.join(PROJECT_ROOT, "data", "raw-files", "raw-tags-french-indexed", "tags.json")
english_path = os.path.join(PROJECT_ROOT, "data", "raw-files", "raw-tags-english-indexed", "tags.json")

def load_tag_mappings():
    with open(arabic_path, 'r', encoding='utf-8') as f:
        arabic_tags = json.load(f)
    with open(french_path, 'r', encoding='utf-8') as f:
        french_tags = json.load(f)
    with open(english_path, 'r', encoding='utf-8') as f:
        english_tags = json.load(f)
    return arabic_tags, french_tags, english_tags

def map_tags(tagged_text):
    arabic_tags, french_tags, english_tags = load_tag_mappings()
    return [
        {
            "word": word,
            "tag": tag,
            "arabic_tag": arabic_tags.get(tag, "Unknown"),
            "french_tag": french_tags.get(tag, "Unknown"),
            "english_tag": english_tags.get(tag, "Unknown")
        }
        for word, tag in tagged_text
    ]