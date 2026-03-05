import re
import requests
from bs4 import BeautifulSoup
from config import KEEP_KEYWORDS, STOP_SECTIONS

def extract_text_from_url(arxiv_id):
    url = f"https://ar5iv.org/html/{arxiv_id}"
    print(f"🌐 Fetching paper from: {url}")
    
    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()  # Raise if 404/500/etc.
    except requests.exceptions.RequestException as e:
        print(f"❌ Fetch failed: {e}")
        return ""  # or raise error, or fallback to PDF
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove unwanted tags
    for tag in soup(['script', 'style', 'figure', 'table', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()
    
    # Get text, preserve some structure
    raw_text = soup.get_text(separator='\n', strip=True)
    
    # Clean up excessive newlines and page artifacts
    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)  # collapse multiple newlines
    raw_text = re.sub(r'Page \d+', '', raw_text)     # remove page numbers
    raw_text = re.sub(r'\[[\d\s]+\]', '', raw_text)  # remove reference [1] style
    
    return raw_text

def filter_sections(text):
    if not text:
        return ""
    
    lines = text.split('\n')
    keep = []
    capturing = False
    
    for line in lines:
        lower = line.lower().strip()
        if any(p in lower for p in KEEP_KEYWORDS) and len(lower) < 100:
            capturing = True
            keep.append(line)
        elif any(s in lower for s in STOP_SECTIONS) and len(lower) < 100:
            capturing = False
        elif capturing and line.strip():  # skip empty lines in captured sections
            keep.append(line)
    
    filtered = '\n'.join(keep)
    # Final cleanup
    filtered = re.sub(r'\n{3,}', '\n\n', filtered)
    
    return filtered

def parse_paper(arxiv_id="1202.2745"):
    print(f"📄 Parsing paper: {arxiv_id}")
    
    raw_text = extract_text_from_url(arxiv_id)
    print(f"✅ Extracted {len(raw_text):,} characters")
    
    if not raw_text:
        print("⚠️ No text extracted — check arXiv ID or internet")
        return ""
    
    filtered = filter_sections(raw_text)
    print(f"✅ Filtered to {len(filtered):,} characters")
    
    return filtered