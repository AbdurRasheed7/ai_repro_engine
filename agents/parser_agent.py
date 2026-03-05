import re
import requests
from bs4 import BeautifulSoup
from config import KEEP_KEYWORDS, STOP_SECTIONS

def extract_text_from_url(arxiv_id):
    url = f"https://ar5iv.org/html/{arxiv_id}"
    print(f"🌐 Fetching paper from: {url}")
    response = requests.get(url, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove unwanted tags
    for tag in soup(['script', 'style', 'figure', 'table']):
        tag.decompose()
    
    return soup.get_text(separator='\n')

def filter_sections(text):
    lines = text.split('\n')
    keep = []
    capturing = False
    
    for line in lines:
        lower = line.lower().strip()
        if any(p in lower for p in KEEP_KEYWORDS) and len(lower) < 80:
            capturing = True
        if any(s in lower for s in STOP_SECTIONS) and len(lower) < 80:
            capturing = False
        if capturing:
            keep.append(line)
    
    return '\n'.join(keep)

def parse_paper(arxiv_id="1202.2745"):
    print(f"📄 Parsing paper: {arxiv_id}")
    raw_text = extract_text_from_url(arxiv_id)
    print(f"✅ Extracted {len(raw_text)} characters")
    
    filtered = filter_sections(raw_text)
    print(f"✅ Filtered to {len(filtered)} characters")
    
    return filtered