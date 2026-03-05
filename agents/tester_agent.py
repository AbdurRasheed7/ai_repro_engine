import json
import os
from datetime import datetime
from config import REPORTS_DIR

def extract_accuracy(output_text):
    """Extract accuracy number from code output"""
    import re
    patterns = [
        r'[Tt]est [Aa]ccuracy[:\s]+([0-9]+\.?[0-9]*)',
        r'[Ff]inal [Aa]ccuracy[:\s]+([0-9]+\.?[0-9]*)',
        r'[Aa]ccuracy[:\s]+([0-9]+\.?[0-9]*)',
        r'[Aa]cc[:\s]+([0-9]+\.?[0-9]*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            value = float(match.group(1))
            # Convert to percentage if decimal
            if value < 1.0:
                value = value * 100
            return round(value, 2)
    return None

def calculate_score(actual, expected, tolerance=2.0):
    """Calculate reproducibility score out of 100"""
    if actual is None:
        return 0
    difference = abs(actual - expected)
    if difference <= tolerance:
        score = 100 - (difference / tolerance * 20)
    else:
        score = max(0, 60 - (difference - tolerance) * 10)
    return round(score, 1)

def run_test(paper_id, stdout, stderr, expected_json_path):
    """Compare results and generate test report"""
    
    # Load expected values
    with open(expected_json_path, 'r') as f:
        expected = json.load(f)

    # Extract actual accuracy
    actual_accuracy = extract_accuracy(stdout)
    expected_accuracy = expected.get('expected_accuracy', 99.2)
    tolerance = expected.get('tolerance', 2.0)

    # Calculate score
    score = calculate_score(actual_accuracy, expected_accuracy, tolerance)

    # Determine status
    if actual_accuracy is None:
        status = "❌ FAIL - No accuracy reported"
    elif abs(actual_accuracy - expected_accuracy) <= tolerance:
        status = "✅ PASS"
    else:
        status = "⚠️ PARTIAL - Outside tolerance"

    # Build result
    result = {
        "paper_id": paper_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expected_accuracy": expected_accuracy,
        "actual_accuracy": actual_accuracy,
        "difference": round(abs((actual_accuracy or 0) - expected_accuracy), 2),
        "tolerance": tolerance,
        "status": status,
        "reproducibility_score": score,
        "has_errors": bool(stderr and "Error" in stderr)
    }

    return result

def generate_html_report(result, code_path):
    """Generate a beautiful HTML report"""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Read generated code
    code_content = ""
    if os.path.exists(code_path):
        with open(code_path, 'r') as f:
            code_content = f.read()

    score = result['reproducibility_score']
    
    # Score color
    if score >= 80:
        score_color = "#2ecc71"
    elif score >= 60:
        score_color = "#f39c12"
    else:
        score_color = "#e74c3c"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Reproducibility Report - {result['paper_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 30px; border-radius: 10px; text-align: center; }}
        .card {{ background: white; padding: 25px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .score {{ font-size: 72px; font-weight: bold; color: {score_color}; text-align: center; }}
        .score-label {{ text-align: center; color: #666; font-size: 18px; }}
        .status {{ font-size: 24px; text-align: center; margin: 10px 0; }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
        .metric-label {{ color: #666; }}
        .metric-value {{ font-weight: bold; }}
        .code-block {{ background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 8px; overflow-x: auto; font-family: monospace; font-size: 13px; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }}
        .pass {{ color: #2ecc71; }}
        .fail {{ color: #e74c3c; }}
        .partial {{ color: #f39c12; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 AI Reproducibility Engine</h1>
        <p>Paper ID: {result['paper_id']} | Generated: {result['timestamp']}</p>
    </div>

    <div class="card">
        <div class="score">{score}</div>
        <div class="score-label">Reproducibility Score / 100</div>
        <div class="status">{result['status']}</div>
    </div>

    <div class="card">
        <h2>📊 Results Comparison</h2>
        <div class="metric">
            <span class="metric-label">Expected Accuracy (Paper Claim)</span>
            <span class="metric-value">{result['expected_accuracy']}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Actual Accuracy (Our Run)</span>
            <span class="metric-value">{result['actual_accuracy']}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Difference</span>
            <span class="metric-value">{result['difference']}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Tolerance</span>
            <span class="metric-value">±{result['tolerance']}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Errors Detected</span>
            <span class="metric-value">{'Yes' if result['has_errors'] else 'No'}</span>
        </div>
    </div>

    <div class="card">
        <h2>💻 Generated Code</h2>
        <div class="code-block">{code_content}</div>
    </div>

    <div class="card" style="text-align:center; color:#666;">
        <p>Generated by AI Reproducibility Engine | Powered by Ollama + qwen2.5-coder</p>
    </div>
</body>
</html>"""

    report_path = os.path.join(REPORTS_DIR, f"{result['paper_id']}_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📄 Report saved: {report_path}")
    return report_path