import os
import sys
from config import OUTPUT_DIR
from agents.parser_agent import parse_paper
from agents.coder_agent import generate_code
from agents.tester_agent import run_test, generate_html_report
from agents.debugger_agent import run_with_debug
from agents.hallucination_agent import analyze_hallucinations, format_hallucination_report
from agents.crew_agents import run_crew_analysis
from agents.domain_detector import detect_domain, format_domain_report, get_code_domain
from data.download_movielens import download_movielens
from agents.rag_agent import get_relevant_context

def save_code(code, paper_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{paper_id}_solution.py")
    with open(output_path, 'w') as f:
        f.write(code)
    print(f"💾 Code saved to: {output_path}")
    return output_path

def run_code(code_path):
    print("🚀 Running generated code...")
    import subprocess
    result = subprocess.run(
        [sys.executable, code_path],
        capture_output=True,
        text=True,
        timeout=600
    )
    print("📊 Output:")
    print(result.stdout)
    if result.stderr and "Error" in result.stderr:
        print("⚠️ Errors:")
        print(result.stderr)
    return result.stdout, result.stderr

def main():
    paper_id = "1807.01622"
    golden_path = f"tests/golden/{paper_id}_expected.json"
    # Step 1 - Parse
    filtered_text = parse_paper(paper_id)
    if not filtered_text.strip():
        print("❌ Could not extract paper content!")
        return

    # Step 2 - Generate code
    # Use RAG to get most relevant context
    rag_context = get_relevant_context(filtered_text)
    # Auto detect domain
    detection = detect_domain(filtered_text)
    print(format_domain_report(detection))
    auto_domain = get_code_domain(detection)
    # Download required datasets
    if detection['domain'] == 'recommendation':
        download_movielens()
    print(f"🎯 Using domain: {auto_domain}")
    code = generate_code(rag_context)
    print("\n--- GENERATED CODE PREVIEW ---")
    print(code[:300])
    print("...")

    # Step 3 - Save code
    code_path = save_code(code, paper_id)

    # Step 4 - Run code
    stdout, stderr, fixed_code, attempts = run_with_debug(code, code_path)
    print(f"🔧 Completed in {attempts} attempt(s)")

    # Step 5 - Test & Score
    print("\n--- REPRODUCIBILITY CHECK ---")
    result = run_test(paper_id, stdout, stderr, golden_path)
    
    print(f"📊 Expected Accuracy : {result['expected_accuracy']}%")
    print(f"📊 Actual Accuracy   : {result['actual_accuracy']}%")
    print(f"📊 Difference        : {result['difference']}%")
    print(f"🏆 Reproducibility Score: {result['reproducibility_score']}/100")
    print(f"✅ Status: {result['status']}")
    # Hallucination Analysis
    hallucination = analyze_hallucinations(code, filtered_text)
    print(format_hallucination_report(hallucination))

    # Step 6 - Generate HTML report
    report_path = generate_html_report(result, code_path)
    
    print(f"\n🌐 Open your report: {report_path}")
   # CrewAI Multi-Agent Analysis (optional - set to True for full demo)
    RUN_CREW = False

    if RUN_CREW:
        crew_result = run_crew_analysis(
            paper_id, filtered_text, rag_context, code,
            stdout, stderr, result, hallucination
        )
    else:
        print("\n⏭️  CrewAI skipped (set RUN_CREW = True to enable)")
        
    print("\n🏁 Pipeline complete!")

if __name__ == "__main__":
    main()