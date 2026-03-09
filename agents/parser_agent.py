import re
import requests
from bs4 import BeautifulSoup
from config import KEEP_KEYWORDS, STOP_SECTIONS


def extract_text_from_url(arxiv_id):
    """Try ar5iv first, fall back to arxiv abstract page"""
    urls = [
        f"https://ar5iv.org/html/{arxiv_id}",
        f"https://arxiv.org/html/{arxiv_id}",   # newer arxiv HTML
    ]

    for url in urls:
        print(f"🌐 Fetching paper from: {url}")
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted tags
            for tag in soup(['script', 'style', 'figure', 'table',
                             'nav', 'header', 'footer', 'aside']):
                tag.decompose()

            raw_text = soup.get_text(separator='\n', strip=True)

            # Clean up artifacts
            raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
            raw_text = re.sub(r'Page \d+', '', raw_text)
            raw_text = re.sub(r'\[[\d\s]+\]', '', raw_text)

            if len(raw_text) > 1000:
                print(f"✅ Extracted {len(raw_text):,} characters")
                return raw_text
            else:
                print(f"⚠️  Too short ({len(raw_text)} chars) — trying next URL...")

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Fetch failed for {url}: {e} — trying next...")

    print("❌ All fetch attempts failed")
    return ""


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
        elif capturing and line.strip():
            keep.append(line)

    filtered = '\n'.join(keep)
    filtered = re.sub(r'\n{3,}', '\n\n', filtered)

    # Guard: if filter returned almost nothing, fall back to full raw text
    # This happens for papers with unusual section names
    if len(filtered) < 500:
        print("⚠️  Section filter returned too little — using full raw text")
        return text

    return filtered


def parse_paper(arxiv_id="1202.2745"):
    print(f"📄 Parsing paper: {arxiv_id}")

    raw_text = extract_text_from_url(arxiv_id)

    if not raw_text:
        print("⚠️ No text extracted — check arXiv ID or internet connection")
        return ""

    filtered = filter_sections(raw_text)
    print(f"✅ Filtered to {len(filtered):,} characters")

    return filtered