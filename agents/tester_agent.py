import json
import os
import re
from datetime import datetime
from config import REPORTS_DIR, TOLERANCE_DEFAULT

def extract_accuracy(output_text):
    """Extract accuracy number from code output - more patterns"""
    patterns = [
        r'[Tt]est [Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Ff]inal [Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Aa]cc[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Tt]op-1 [Aa]ccuracy[:\s=]+([0-9]+\.?[0-9]*)',
        r'[Aa]ccuracy\s*=\s*([0-9]+\.?[0-9]*)',
        r'accuracy\s*[:=]\s*([0-9]+\.?[0-9]*)',
        r'Accuracy:\s*([0-9]+\.?[0-9]*)%',
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if value < 1.0:
                value *= 100
            return round(value, 2)
    return None

def calculate_score(actual, expected, tolerance=2.0):
    """Calculate reproducibility score out of 100"""
    if actual is None:
        return 0
    difference = abs(actual - expected)
    if difference <= tolerance:
        # Perfect or close: high score
        score = 100 - (difference / tolerance * 20)
    else:
        # Larger difference: steeper drop
        score = max(0, 80 - (difference - tolerance) * 15)
    return round(score, 1)

def run_test(paper_id, stdout, stderr, expected_json_path):
    """Compare results and generate test report"""

    try:
        # Load expected values
        with open(expected_json_path, 'r') as f:
            expected = json.load(f)
    except FileNotFoundError:
        print(f"   Warning: No golden JSON found for {paper_id}. Using defaults.")
        expected = {"expected_accuracy": None, "tolerance": TOLERANCE_DEFAULT}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON: {expected_json_path}")
        expected = {"expected_accuracy": None, "tolerance": TOLERANCE_DEFAULT}

    # Extract actual accuracy
    actual_accuracy = extract_accuracy(stdout)

    expected_accuracy = expected.get('expected_accuracy')
    tolerance = expected.get('tolerance', 2.0)

    # Calculate score
    if expected_accuracy is not None:
        # Golden JSON exists — score against expected
        score = calculate_score(actual_accuracy, expected_accuracy, tolerance)
    elif actual_accuracy is not None:
        # No golden JSON but code ran and produced accuracy — give execution score
        # Base score: 60 for running, up to 85 based on accuracy level
        if actual_accuracy >= 90:
            score = 85
        elif actual_accuracy >= 80:
            score = 75
        elif actual_accuracy >= 70:
            score = 65
        else:
            score = 60
    else:
        # No golden JSON and no accuracy — complete failure
        score = 0

    # Determine status
    if actual_accuracy is None:
        status = "❌ FAIL - No accuracy reported"
    elif expected_accuracy is None:
        status = f"✅ PASS - Executed successfully (no baseline to compare)"
    elif abs(actual_accuracy - expected_accuracy) <= tolerance:
        status = "✅ PASS"
    else:
        status = "⚠️ PARTIAL - Outside tolerance"

    # Build result — ALL keys always present with safe defaults
    result = {
        "paper_id":             paper_id,
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
        "expected_accuracy":    expected_accuracy,
        "actual_accuracy":      actual_accuracy,
        "difference":           round(abs((actual_accuracy or 0) - (expected_accuracy or 0)), 2)
                                if actual_accuracy is not None and expected_accuracy is not None else None,
        "tolerance":            tolerance,
        "status":               status,
        "reproducibility_score": score,
        "has_errors":           bool(stderr and ("Error" in stderr or "Exception" in stderr)),
        "stderr_preview":       stderr[:200] if stderr else None
    }

    return result

def generate_html_report(result, code_path):
    """Generate a beautiful HTML report — safe against missing keys"""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Read generated code
    code_content = "Code file not found"
    if os.path.exists(code_path):
        try:
            with open(code_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as e:
            code_content = f"Error reading code: {e}"

    # Safe key access with fallbacks for every field
    paper_id            = result.get('paper_id',             'Unknown')
    timestamp           = result.get('timestamp',            'N/A')
    score               = result.get('reproducibility_score', 0)
    status              = result.get('status',               'FAILED')
    expected_accuracy   = result.get('expected_accuracy',    None)
    actual_accuracy     = result.get('actual_accuracy',      None)
    difference          = result.get('difference',           None)
    tolerance           = result.get('tolerance',            2.0)
    has_errors          = result.get('has_errors',           False)

    # Score color
    if score >= 80:
        score_color = "#2ecc71"   # green
    elif score >= 60:
        score_color = "#f39c12"   # orange
    else:
        score_color = "#e74c3c"   # red

    # Status CSS class
    if 'PASS' in status:
        status_class = 'pass'
    elif 'FAIL' in status:
        status_class = 'fail'
    else:
        status_class = 'partial'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Reproducibility Report - {paper_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1000px; margin: 40px auto; padding: 20px; background: #f8f9fa; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 40px 30px; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.15); }}
        .card {{ background: white; padding: 30px; margin: 25px 0; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }}
        .score {{ font-size: 90px; font-weight: bold; color: {score_color}; text-align: center; margin: 0; }}
        .score-label {{ text-align: center; color: #555; font-size: 22px; margin-bottom: 20px; }}
        .status {{ font-size: 28px; text-align: center; margin: 15px 0; font-weight: bold; }}
        .metric {{ display: flex; justify-content: space-between; padding: 15px 0; border-bottom: 1px solid #eee; font-size: 18px; }}
        .metric-label {{ color: #555; font-weight: 500; }}
        .metric-value {{ font-weight: bold; }}
        .code-block {{ background: #1e272e; color: #c3d2df; padding: 25px; border-radius: 10px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 14px; white-space: pre-wrap; max-height: 500px; overflow-y: auto; margin-top: 15px; }}
        .pass {{ color: #2ecc71; }}
        .fail {{ color: #e74c3c; }}
        .partial {{ color: #f39c12; }}
        .footer {{ text-align: center; color: #777; margin-top: 40px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 AI Reproducibility Engine Report</h1>
        <p>Paper ID: <strong>{paper_id}</strong> | Generated: {timestamp}</p>
    </div>

    <div class="card">
        <div class="score">{score}</div>
        <div class="score-label">Reproducibility Score / 100</div>
        <div class="status {status_class}">{status}</div>
    </div>

    <div class="card">
        <h2>📊 Results Comparison</h2>
        <div class="metric">
            <span class="metric-label">Expected Accuracy (Paper Claim)</span>
            <span class="metric-value">{f"{expected_accuracy}%" if expected_accuracy is not None else "N/A"}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Actual Accuracy (Our Run)</span>
            <span class="metric-value">{f"{actual_accuracy}%" if actual_accuracy is not None else "N/A"}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Difference</span>
            <span class="metric-value">{f"{difference}%" if difference is not None else "N/A"}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Tolerance</span>
            <span class="metric-value">±{tolerance}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Runtime Errors</span>
            <span class="metric-value">{'Yes' if has_errors else 'No'}</span>
        </div>
    </div>

    <div class="card">
        <h2>💻 Generated Code</h2>
        <div class="code-block">{code_content.replace('<', '&lt;').replace('>', '&gt;')}</div>
    </div>

    <div class="footer">
        Generated by AI Reproducibility Engine | Powered by Groq + Llama 3.3 70B | March 2026
    </div>
</body>
</html>"""

    report_path = os.path.join(REPORTS_DIR, f"{paper_id}_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"📄 Report saved: {report_path}")
    return report_path