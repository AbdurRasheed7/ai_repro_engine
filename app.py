import streamlit as st
import os
import sys
import json
import subprocess
from datetime import datetime

# Add project to path
sys.path.append(os.path.dirname(__file__))

from agents.parser_agent import parse_paper
from agents.coder_agent import generate_code
from agents.tester_agent import run_test, generate_html_report
from agents.rag_agent import get_relevant_context

# Page config
st.set_page_config(
    page_title="AI Reproducibility Engine",
    page_icon="🔬",
    layout="centered"
)

# Header
st.markdown("""
    <div style='text-align: center; padding: 20px; background: #2c3e50; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white;'>🔬 AI Reproducibility Engine</h1>
        <p style='color: #bdc3c7;'>From Research Paper to Executable Code</p>
    </div>
""", unsafe_allow_html=True)

# Input section
st.markdown("### 📄 Enter ArXiv Paper ID")
col1, col2 = st.columns([3, 1])

with col1:
    paper_id = st.text_input(
        "ArXiv ID",
        placeholder="e.g. 1202.2745",
        label_visibility="collapsed"
    )

with col2:
    run_button = st.button("🚀 Run", use_container_width=True)

# Example papers
st.markdown("**Quick test papers:**")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("1202.2745", use_container_width=True):
        paper_id = "1202.2745"
        run_button = True
with c2:
    if st.button("1807.01622", use_container_width=True):
        paper_id = "1807.01622"
        run_button = True
with c3:
    if st.button("1409.1556", use_container_width=True):
        paper_id = "1409.1556"
        run_button = True

st.markdown("**Algorithm papers:**")
a1, a2 = st.columns(2)
with a1:
    if st.button("2301.07041", use_container_width=True):
        paper_id = "2301.07041"
        run_button = True
with a2:
    if st.button("1811.10154", use_container_width=True):
        paper_id = "1811.10154"
        run_button = True

# Settings
with st.expander("⚙️ Settings"):
    domain = st.selectbox(
        "Paper Domain",
        ["ml", "algorithm"],
        format_func=lambda x: "🧠 ML Research Paper" if x == "ml" else "⚙️ Algorithm Paper"
    )
    expected_accuracy = st.slider("Expected Accuracy (%)", 80.0, 100.0, 99.0, 0.1)
    tolerance = st.slider("Tolerance (%)", 0.5, 5.0, 2.0, 0.5)

# Run pipeline
if run_button and paper_id:
    st.markdown("---")
    
    # Progress bar
    progress = st.progress(0)
    status = st.empty()

    try:
        # Step 1 - Parse
        status.info("📄 Step 1/4 — Fetching and parsing paper...")
        progress.progress(10)
        filtered_text = parse_paper(paper_id)
        
        if not filtered_text.strip():
            st.error("❌ Could not extract content from this paper!")
            st.stop()
        
        st.success(f"✅ Extracted {len(filtered_text)} characters")
        progress.progress(25)

        # Step 2 - Generate code
        status.info("🤖 Step 2/4 — Generating code with Ollama AI...")
        progress.progress(30)
        rag_context = get_relevant_context(filtered_text)
        code = generate_code(rag_context, domain=domain)
        st.success("✅ Code generated successfully!")
        progress.progress(50)

        # Step 3 - Save and run code
        status.info("🚀 Step 3/4 — Running generated code...")
        progress.progress(55)
        
        os.makedirs("generated_code", exist_ok=True)
        code_path = f"generated_code/{paper_id}_solution.py"
        with open(code_path, 'w') as f:
            f.write(code)

        import subprocess
        result = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=600
        )
        stdout = result.stdout
        stderr = result.stderr
        progress.progress(80)

        # Step 4 - Score
        status.info("📊 Step 4/4 — Calculating reproducibility score...")
        
        # Create temp golden file
        golden = {
            "paper_id": paper_id,
            "expected_accuracy": expected_accuracy,
            "tolerance": tolerance,
            "seed": 42,
            "epochs": 5,
            "notes": "User submitted paper"
        }
        golden_path = f"tests/golden/{paper_id}_expected.json"
        os.makedirs("tests/golden", exist_ok=True)
        with open(golden_path, 'w') as f:
            json.dump(golden, f)

        test_result = run_test(paper_id, stdout, stderr, golden_path)
        report_path = generate_html_report(test_result, code_path)
        progress.progress(100)
        status.empty()

        # Results
        st.markdown("---")
        st.markdown("## 📊 Results")

        # Score display
        score = test_result['reproducibility_score']
        if score >= 80:
            color = "🟢"
        elif score >= 60:
            color = "🟡"
        else:
            color = "🔴"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Score", f"{score}/100")
        with col2:
            st.metric("🎯 Actual Accuracy", f"{test_result['actual_accuracy']}%")
        with col3:
            st.metric("📏 Difference", f"{test_result['difference']}%")

        # Status
        st.markdown(f"### {color} {test_result['status']}")

        # Code preview
        with st.expander("💻 View Generated Code"):
            st.code(code, language='python')

        # Output preview
        if stdout:
            with st.expander("📄 Training Output"):
                st.text(stdout)

        # Report link
        st.markdown("---")
        st.success(f"📄 Full report saved to: `{report_path}`")
        st.info("Open the HTML file in your browser for the full report!")

    except subprocess.TimeoutExpired:
        st.error("⏱️ Code took too long to run (>5 mins). Try a simpler paper.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

elif run_button and not paper_id:
    st.warning("⚠️ Please enter an ArXiv paper ID first!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666;'>Powered by Ollama + qwen2.5-coder | Built by Abdur Rasheed</p>",
    unsafe_allow_html=True
)