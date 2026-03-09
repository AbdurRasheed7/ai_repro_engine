import json
import os
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from config import GOLDEN_DIR, GROQ_MODEL

load_dotenv()

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.0,       # zero temp — we want deterministic extraction
    max_tokens=256,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def extract_expected_accuracy(paper_id, filtered_text):
    """
    Use Groq to extract the best claimed accuracy from a paper.
    Saves result to tests/golden/{paper_id}_expected.json.
    Returns the golden JSON path.
    """
    golden_path = os.path.join(GOLDEN_DIR, f"{paper_id}_expected.json")

    # Already exists — skip extraction
    if os.path.exists(golden_path):
        print(f"   📋 Found existing golden JSON: {golden_path}")
        return golden_path

    print(f"   🔍 No golden JSON found — extracting expected accuracy from paper...")

    prompt = f"""Read this research paper excerpt carefully.
Extract the BEST or FINAL accuracy the authors claim to achieve on their main task.

RULES:
- Return ONLY a JSON object, nothing else, no markdown, no explanation
- Format: {{"expected_accuracy": 99.0, "tolerance": 2.0}}
- expected_accuracy must be a float percentage (e.g. 99.0, not 0.99)
- Use tolerance 2.0 for precise claims, 5.0 for approximate or cross-domain results
- If multiple accuracies mentioned, pick the BEST one they claim
- If no accuracy found anywhere, return: {{"expected_accuracy": null, "tolerance": 2.0}}

PAPER EXCERPT:
{filtered_text[:4000]}"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()

        # Clean markdown fences if Groq adds them anyway
        if "```" in text:
            text = re.sub(r"```[a-z]*", "", text).replace("```", "").strip()

        data = json.loads(text)

        # Validate structure
        if "expected_accuracy" not in data:
            data = {"expected_accuracy": None, "tolerance": 2.0}

        # Ensure types are correct
        if data["expected_accuracy"] is not None:
            data["expected_accuracy"] = float(data["expected_accuracy"])
            # If Groq returned 0.99 instead of 99.0 — fix it
            if data["expected_accuracy"] < 1.0:
                data["expected_accuracy"] = round(data["expected_accuracy"] * 100, 2)
            data["tolerance"] = float(data.get("tolerance", 2.0))

        # Save to golden directory
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        acc = data.get("expected_accuracy")
        tol = data.get("tolerance", 2.0)
        if acc is not None:
            print(f"   📋 Extracted expected accuracy: {acc}% ± {tol}% → saved to golden JSON")
        else:
            print(f"   ⚠️  Could not extract accuracy from paper — will score by execution only")

        return golden_path

    except json.JSONDecodeError as e:
        print(f"   ⚠️  Groq returned invalid JSON: {e} — saving null golden")
        return _save_null_golden(paper_id, golden_path)

    except Exception as e:
        print(f"   ⚠️  Golden extraction failed: {e} — saving null golden")
        return _save_null_golden(paper_id, golden_path)


def _save_null_golden(paper_id, golden_path):
    """Save a null golden JSON so pipeline doesn't retry on next run"""
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump({"expected_accuracy": None, "tolerance": 2.0}, f, indent=2)
    return golden_path