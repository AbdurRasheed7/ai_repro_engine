import streamlit as st
import os
import sys
import json
import subprocess
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(__file__))

from agents.parser_agent import parse_paper
from agents.rag_agent import get_relevant_context
from agents.coder_agent import generate_code
from agents.domain_detector import detect_domain, format_domain_report, get_code_domain  # ← add these
from agents.tester_agent import run_test, generate_html_report

# Page config
st.set_page_config(
    page_title="AI Reproducibility Engine",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .header {background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 30px;}
        .metric-box {background: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;}
        .success {color: #2ecc71; font-weight: bold;}
        .error {color: #e74c3c; font-weight: bold;}
        .code {background: #1e272e; color: #c3d2df; padding: 15px; border-radius: 8px; overflow-x: auto; font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>🔬 AI Reproducibility Engine</h1>
        <p>From Research Paper PDF/ArXiv → Verified Executable Code</p>
        <small>Powered by Groq + Llama 3.3 70B | Built by Mohammed Abdur Rasheed</small>
    </div>
""", unsafe_allow_html=True)

# Main input
st.subheader("📄 Enter ArXiv Paper ID")
col1, col2, col3 = st.columns([4, 2, 2])

with col1:
    paper_id = st.text_input("ArXiv ID", placeholder="e.g. 1512.03385 (ResNet)", label_visibility="collapsed")

with col2:
    run_button = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

with col3:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# Quick test buttons
st.subheader("Quick Test Papers")
cols = st.columns(5)
papers = [
    ("1512.03385", "ResNet"),
    ("1409.1556", "VGG"),
    ("1202.2745", "Multi-Column CNN"),
    ("1807.01622", "Neural Processes"),
    ("2301.07041", "Homomorphic Enc")
]
for i, (pid, label) in enumerate(papers):
    with cols[i]:
        if st.button(pid, help=label, use_container_width=True):
            paper_id = pid
            run_button = True

# Settings expander
with st.expander("⚙️ Advanced Settings", expanded=False):
    domain_override = st.selectbox(
        "Force Domain",
        ["Auto-detect", "ml (CV/ML)", "algorithm", "nlp", "recommendation", "rl", "graph"],
        index=0
    )
    expected_acc = st.slider("Expected Accuracy (%)", 70.0, 100.0, 95.0, 0.1)
    tolerance = st.slider("Tolerance (±%)", 0.5, 5.0, 2.0, 0.1)

# Run the pipeline when button pressed
if run_button and paper_id:
    st.markdown("---")
    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Processing paper..."):
        try:
            # Step 1: Parse
            status_text.info("Step 1/5: Fetching & parsing paper...")
            progress_bar.progress(10)
            filtered_text = parse_paper(paper_id)
            if not filtered_text.strip():
                raise ValueError("No text extracted from paper.")
            st.success(f"Extracted {len(filtered_text):,} chars")

            # Domain detection
            status_text.info("Step 2/5: Detecting domain...")
            progress_bar.progress(25)
            domain_detection = detect_domain(filtered_text)
            st.info(format_domain_report(domain_detection))
            domain = get_code_domain(domain_detection) if domain_override == "Auto-detect" else domain_override.split()[0]

            # Step 3: RAG
            status_text.info("Step 3/5: Building RAG context...")
            progress_bar.progress(40)
            rag_context = get_relevant_context(filtered_text)

            # Step 4: Generate code with Groq
            status_text.info("Step 4/5: Generating code with Groq...")
            progress_bar.progress(60)
            code = generate_code(rag_context, domain=domain)

            # Save code
            os.makedirs("generated_code", exist_ok=True)
            code_path = f"generated_code/{paper_id}_solution.py"
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code)

            st.success("Code generated!")
            with st.expander("View Generated Code"):
                st.code(code, language="python")

            # Step 5: Run & Test
            status_text.info("Step 5/5: Running code & verifying...")
            progress_bar.progress(80)

            result = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                text=True,
                timeout=900  # 15 min
            )
            stdout = result.stdout
            stderr = result.stderr

            # Create temp golden
            golden = {
                "expected_accuracy": expected_acc,
                "tolerance": tolerance
            }
            golden_path = f"tests/golden/{paper_id}_temp.json"
            os.makedirs("tests/golden", exist_ok=True)
            with open(golden_path, 'w') as f:
                json.dump(golden, f)

            test_result = run_test(paper_id, stdout, stderr, golden_path)
            report_path = generate_html_report(test_result, code_path)

            progress_bar.progress(100)
            status_text.success("Pipeline complete!")

            # Display results
            st.markdown("## Results")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Reproducibility Score", f"{test_result['reproducibility_score']}/100")
            with cols[1]:
                st.metric("Actual Accuracy", f"{test_result['actual_accuracy'] or 'N/A'}%")
            with cols[2]:
                st.metric("Difference", f"{test_result['difference'] or 'N/A'}%")

            st.markdown(f"### {test_result['status']}")

            if stdout:
                with st.expander("Training Logs"):
                    st.text(stdout)

            if stderr:
                with st.expander("Errors/Warnings"):
                    st.error(stderr)

            # Report preview
            st.markdown("## Full Report")
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)

            st.success(f"Report saved to: `{report_path}`")
            st.download_button("Download HTML Report", html_content, file_name=f"{paper_id}_report.html")

        except subprocess.TimeoutExpired:
            st.error("⏱️ Code execution timed out (>15 min). Try a lighter paper.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    if run_button:
        st.warning("Please enter an ArXiv ID first!")

# Footer
st.markdown("---")
st.caption("AI Reproducibility Engine v1 • Powered by Groq • Built in Hyderabad • 2026")