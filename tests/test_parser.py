import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents.parser_agent import parse_paper

arxiv_id = "1202.2745"  # smoke test
filtered = parse_paper(arxiv_id)
print("\n=== Filtered Preview (first 800 chars) ===\n")
print(filtered[:800])
print(f"\nTotal chars: {len(filtered)}")
print("Success if Methodology/Experiments content appears.")