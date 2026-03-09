import streamlit as st
import os
import sys
import json
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(__file__))

from agents.parser_agent import parse_paper
from agents.rag_agent import get_relevant_context
from agents.coder_agent import generate_code
from agents.domain_detector import detect_domain, format_domain_report, get_code_domain
from agents.tester_agent import run_test, generate_html_report
from agents.golden_agent import extract_expected_accuracy
from agents.hallucination_agent import analyze_hallucinations, format_hallucination_report
from utils.docker_helper import run_code_in_docker
from agents.debugger_agent import run_with_debug
from config import OUTPUT_DIR, REPORTS_DIR, GOLDEN_DIR, CODE_TIMEOUT_SEC, TOLERANCE_DEFAULT

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Reproducibility Engine",
    page_icon="🔬",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
    <style>
        .header {background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 30px;}
        .metric-box {background: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;}
        .success {color: #2ecc71; font-weight: bold;}
        .error {color: #e74c3c; font-weight: bold;}
        .code {background: #1e272e; color: #c3d2df; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
    <div class="header">
        <h1>🔬 AI Reproducibility Engine</h1>
        <p>From Research Paper PDF/ArXiv → Verified Executable Code</p>
        <small>Powered by Groq + Llama 3.3 70B | Built by Mohammed Abdur Rasheed</small>
    </div>
""", unsafe_allow_html=True)

# ── Main input ─────────────────────────────────────────────
st.subheader("📄 Enter ArXiv Paper ID")
col1, col2, col3 = st.columns([4, 2, 2])

with col1:
    paper_id = st.text_input("ArXiv ID", placeholder="e.g. 1512.03385 (ResNet)", label_visibility="collapsed")

with col2:
    run_button = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

with col3:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# ── Quick test buttons ─────────────────────────────────────
st.subheader("Quick Test Papers")
cols = st.columns(5)
papers = [
    ("1512.03385", "ResNet"),
    ("1409.1556",  "VGG"),
    ("1202.2745",  "Multi-Column CNN"),
    ("1706.03762", "Transformer"),
    ("2301.07041", "Homomorphic Enc"),
]
for i, (pid, label) in enumerate(papers):
    with cols[i]:
        if st.button(pid, help=label, use_container_width=True):
            paper_id = pid
            run_button = True

# ── Advanced settings ──────────────────────────────────────
with st.expander("⚙️ Advanced Settings", expanded=False):
    domain_override = st.selectbox(
        "Force Domain",
        ["Auto-detect", "ml", "algorithm", "nlp", "recommendation", "rl", "graph"],
        index=0
    )
    use_docker = st.checkbox("Run in Docker (isolated, recommended)", value=True)
    use_golden = st.checkbox("Auto-extract expected accuracy from paper", value=True)
    col_a, col_b = st.columns(2)
    with col_a:
        manual_expected_acc = st.slider("Manual Expected Accuracy (%)", 70.0, 100.0, 95.0, 0.1,
                                         disabled=use_golden)
    with col_b:
        tolerance = st.slider("Tolerance (±%)", 0.5, 5.0, TOLERANCE_DEFAULT, 0.1)

# ── Pipeline ───────────────────────────────────────────────
if run_button and paper_id:
    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Parse
        status_text.info("Step 1/6: Fetching & parsing paper...")
        progress_bar.progress(10)
        filtered_text = parse_paper(paper_id)
        if not filtered_text.strip():
            raise ValueError("No text extracted from paper.")
        st.success(f"✅ Extracted {len(filtered_text):,} chars")

        # Step 2: Domain detection
        status_text.info("Step 2/6: Detecting domain...")
        progress_bar.progress(20)
        domain_detection = detect_domain(filtered_text)
        st.info(format_domain_report(domain_detection))
        domain = get_code_domain(domain_detection) if domain_override == "Auto-detect" else domain_override

        # Step 3: RAG context
        status_text.info("Step 3/6: Building RAG context...")
        progress_bar.progress(35)
        rag_context = get_relevant_context(filtered_text, domain=domain)
        st.success(f"✅ Retrieved {len(rag_context):,} chars of context")

        # Step 4: Generate code
        status_text.info("Step 4/6: Generating code with Groq...")
        progress_bar.progress(50)
        code = generate_code(rag_context, domain=domain)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        code_path = os.path.join(OUTPUT_DIR, f"{paper_id}_solution.py")
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)

        st.success("✅ Code generated!")
        with st.expander("View Generated Code"):
            st.code(code, language="python")

        # Step 5: Execute code
        status_text.info("Step 5/6: Running code & verifying...")
        progress_bar.progress(70)

        if use_docker:
            success, logs, error = run_code_in_docker(code, paper_id)
            if success:
                stdout = logs
                stderr = ""
                final_code = code
                st.success("✅ Docker execution succeeded!")
            else:
                st.warning("⚠️ Docker failed — falling back to local debugger...")
                stdout, stderr, final_code, attempts = run_with_debug(code, code_path, domain=domain)
                # Save final (possibly fixed) code
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(final_code)
                if "Final Accuracy" in stdout:
                    st.success(f"✅ Debugger recovered after {attempts} attempt(s)!")
                else:
                    st.error(f"❌ Execution failed after {attempts} attempt(s)")
        else:
            # Local execution only
            stdout, stderr, final_code, attempts = run_with_debug(code, code_path, domain=domain)
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(final_code)
            if "Final Accuracy" in stdout:
                st.success(f"✅ Ran locally in {attempts} attempt(s)!")
            else:
                st.error(f"❌ Local execution failed after {attempts} attempt(s)")

        # Step 6: Scoring & report
        status_text.info("Step 6/6: Scoring & generating report...")
        progress_bar.progress(88)

        # Golden JSON: auto-extract from paper or use manual slider
        if use_golden:
            golden_path = extract_expected_accuracy(paper_id, filtered_text)
        else:
            os.makedirs(GOLDEN_DIR, exist_ok=True)
            golden_path = os.path.join(GOLDEN_DIR, f"{paper_id}_manual.json")
            with open(golden_path, 'w') as f:
                json.dump({"expected_accuracy": manual_expected_acc, "tolerance": tolerance}, f)

        test_result = run_test(paper_id, stdout, stderr, golden_path)

        # Hallucination analysis
        hallucination = analyze_hallucinations(final_code, filtered_text)

        report_path = generate_html_report(test_result, code_path)
        progress_bar.progress(100)
        status_text.success("✅ Pipeline complete!")

        # ── Display results ────────────────────────────────
        st.markdown("## 📊 Results")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Reproducibility Score",
                      f"{test_result.get('reproducibility_score', 0)}/100")
        with c2:
            st.metric("Expected Accuracy",
                      f"{test_result.get('expected_accuracy') or 'N/A'}%")
        with c3:
            st.metric("Actual Accuracy",
                      f"{test_result.get('actual_accuracy') or 'N/A'}%")
        with c4:
            st.metric("Difference",
                      f"{test_result.get('difference') or 'N/A'}%")

        st.markdown(f"### {test_result.get('status', 'FAILED')}")

        # Hallucination score
        st.markdown("## 🧠 Hallucination Analysis")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Hallucination Score",
                      f"{hallucination.get('hallucination_score', 0)}/100")
        with h2:
            st.metric("Values from Paper",
                      hallucination.get('total_from_paper', 0))
        with h3:
            st.metric("AI Assumptions",
                      hallucination.get('total_assumptions', 0))

        with st.expander("Hallucination Details"):
            st.text(format_hallucination_report(hallucination))

        # Logs
        if stdout:
            with st.expander("📋 Training Logs"):
                st.text(stdout)

        if stderr:
            with st.expander("⚠️ Errors / Warnings"):
                st.error(stderr[:2000])

        # Full HTML report
        st.markdown("## 📄 Full Report")
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)

        st.success(f"Report saved to: `{report_path}`")
        st.download_button(
            "⬇️ Download HTML Report",
            html_content,
            file_name=f"{paper_id}_report.html",
            mime="text/html"
        )

    except Exception as e:
        st.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

elif run_button:
    st.warning("Please enter an ArXiv ID first!")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("AI Reproducibility Engine v1 • Powered by Groq • Built in Hyderabad • 2026")