import os
import sys
import argparse
from datetime import datetime

from config import OUTPUT_DIR, REPORTS_DIR, RANDOM_SEED
from agents.parser_agent import parse_paper
from agents.rag_agent import get_relevant_context
from agents.coder_agent import generate_code
from agents.domain_detector import detect_domain, format_domain_report, get_code_domain
from agents.debugger_agent import run_with_debug
from agents.tester_agent import run_test, generate_html_report
from agents.hallucination_agent import analyze_hallucinations, format_hallucination_report
from agents.crew_agents import run_crew_analysis
from data.download_movielens import download_movielens
from utils.docker_helper import run_code_in_docker

def save_code(code, paper_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    code_path = os.path.join(OUTPUT_DIR, f"{paper_id}_solution.py")
    with open(code_path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"💾 Code saved: {code_path}")
    return code_path

def main(paper_id="1512.03385", run_crew=False, force_domain=None):
    print(f"\n{'='*60}")
    print(f"🚀 Starting Reproducibility Pipeline for paper: {paper_id}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"{'='*60}\n")

    try:
        # Step 1: Parse paper
        print("Step 1: Fetching & parsing paper...")
        filtered_text = parse_paper(paper_id)
        if not filtered_text.strip():
            raise ValueError("No meaningful text extracted from paper.")
        print(f"   Extracted {len(filtered_text):,} chars")

        # Step 2: Domain detection
        print("Step 2: Detecting domain...")
        detection = detect_domain(filtered_text)
        print(format_domain_report(detection))
        domain = get_code_domain(detection)
        if force_domain:
            domain = force_domain
            print(f"   Overridden domain: {domain}")
        print(f"   Final domain for code gen: {domain}")

        # Download dataset if needed
        if detection['domain'] == 'recommendation':
            print("   Downloading MovieLens dataset...")
            download_movielens()

        # Step 3: RAG context
        print("Step 3: Building RAG context...")
        rag_context = get_relevant_context(filtered_text)
        print(f"   Retrieved {len(rag_context):,} chars of relevant context")

        # Step 4: Generate code
        print("Step 4: Generating code with Groq...")
        code = generate_code(rag_context, domain=domain)
        print("\n--- GENERATED CODE PREVIEW (first 500 chars) ---")
        print(code[:500] + "..." if len(code) > 500 else code)

        # Step 5: Save & Debug/Run code
        print("\nStep 5: Saving & executing code...")
        code_path = save_code(code, paper_id)
        stdout, stderr, final_code, attempts = run_with_debug(code, code_path)
        print(f"   Completed in {attempts} attempt(s)")
        if stdout:
            print("\n--- STDOUT PREVIEW ---")
            print(stdout[:800] + "..." if len(stdout) > 800 else stdout)
        if stderr:
            print("\n--- STDERR PREVIEW ---")
            print(stderr[:800] + "..." if len(stderr) > 800 else stderr)

        # Step 6: Reproducibility test
        print("\nStep 6: Reproducibility check...")
        golden_path = f"tests/golden/{paper_id}_expected.json"
        if not os.path.exists(golden_path):
            print(f"   Warning: No golden JSON found for {paper_id}. Using defaults.")
            golden_path = None  # or create temp one in tester_agent

        test_result = run_test(paper_id, stdout, stderr, golden_path) if golden_path else {
            "reproducibility_score": 0,
            "status": "⚠️ No expected values",
            "actual_accuracy": None,
            "expected_accuracy": None,
            "difference": None
        }

        print(f"   Expected Accuracy : {test_result.get('expected_accuracy', 'N/A')}%")
        print(f"   Actual Accuracy   : {test_result.get('actual_accuracy', 'N/A')}%")
        print(f"   Difference        : {test_result.get('difference', 'N/A')}%")
        print(f"   🏆 Reproducibility Score: {test_result['reproducibility_score']}/100")
        print(f"   Status: {test_result['status']}")

        # Step 7: Hallucination analysis
        print("\nStep 7: Hallucination check...")
        hallucination = analyze_hallucinations(final_code or code, filtered_text)
        print(format_hallucination_report(hallucination))

        # Step 8: HTML report
        print("\nStep 8: Generating HTML report...")
        report_path = generate_html_report(test_result, code_path)
        print(f"   Report saved: {report_path}")
        print(f"   Open in browser: file://{os.path.abspath(report_path)}")

        # Optional: Full CrewAI multi-agent review
        if run_crew:
            print("\nStep 9: Running CrewAI multi-agent analysis...")
            crew_result = run_crew_analysis(
                paper_id, filtered_text, rag_context, final_code or code,
                stdout, stderr, test_result, hallucination
            )
            with open(f"reports/{paper_id}_crew_summary.txt", "w") as f:
                f.write(str(crew_result))
            print("CrewAI summary saved to reports/")
            print("\nCrewAI Analysis Summary:")
            print(crew_result)
        else:
            print("\n⏭️ CrewAI skipped (use --crew to enable)")

        print("\n🏁 Pipeline complete!")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Reproducibility Engine Pipeline")
    parser.add_argument("--paper", default="1512.03385", help="ArXiv paper ID")
    parser.add_argument("--crew", action="store_true", help="Run full CrewAI analysis")
    parser.add_argument("--domain", help="Force domain (ml, algorithm, nlp, etc.)")
    
    args = parser.parse_args()
    
    main(
        paper_id=args.paper,
        run_crew=args.crew,
        force_domain=args.domain
    )