import re

# Things that should come from paper
PAPER_INDICATORS = [
    "according to", "paper states", "authors use", "proposed",
    "we use", "we propose", "architecture consists", "trained on",
    "dataset contains", "accuracy of", "learning rate of"
]

# Things AI likely assumed
ASSUMPTION_INDICATORS = [
    "UNKNOWN", "assumed", "default", "typical", "standard",
    "common", "usually", "generally", "placeholder"
]

# Known assumption patterns in code
ASSUMPTION_PATTERNS = [
    (r"learning_rate\s*=\s*[\d.]+", "Learning rate value"),
    (r"batch_size\s*=\s*[\d]+", "Batch size value"),
    (r"num_epochs\s*=\s*[\d]+", "Number of epochs"),
    (r"hidden_size\s*=\s*[\d]+", "Hidden layer size"),
    (r"dropout\s*=\s*[\d.]+", "Dropout rate"),
    (r"momentum\s*=\s*[\d.]+", "Momentum value"),
    (r"weight_decay\s*=\s*[\d.]+", "Weight decay value"),
    (r"nn\.Linear\([\d,\s]+\)", "Fully connected layer size"),
    (r"nn\.Conv2d\([\d,\s]+\)", "Convolution layer parameters"),
    (r"kernel_size\s*=\s*[\d]+", "Kernel size"),
    (r"padding\s*=\s*[\d]+", "Padding value"),
    (r"stride\s*=\s*[\d]+", "Stride value"),
    (r"optim\.Adam", "Optimizer type (Adam assumed)"),
    (r"optim\.SGD", "Optimizer type (SGD assumed)"),
    (r"optim\.RMSprop", "Optimizer type (RMSprop assumed)"),
]

def analyze_hallucinations(code, paper_text):
    """Analyze generated code for hallucinations and assumptions"""
    
    flags = []
    assumptions = []
    from_paper = []

    # Check for explicit UNKNOWN comments
    unknown_lines = [l.strip() for l in code.split('\n') if 'UNKNOWN' in l]
    for line in unknown_lines:
        flags.append({
            "type": "EXPLICIT_ASSUMPTION",
            "severity": "HIGH",
            "detail": line,
            "message": "AI explicitly flagged this as unknown"
        })

    # Check for assumption patterns in code
    for pattern, description in ASSUMPTION_PATTERNS:
        matches = re.findall(pattern, code)
        if matches:
            # Check if this value is mentioned in paper
            value = matches[0]
            number = re.findall(r'[\d.]+', value)
            
            found_in_paper = False
            if number:
                for n in number:
                    if n in paper_text and len(n) > 1:
                        found_in_paper = True
                        break
            
            if found_in_paper:
                from_paper.append({
                    "detail": f"{description}: {value}",
                    "message": "Value found in paper ✅"
                })
            else:
                assumptions.append({
                    "type": "ASSUMED_VALUE",
                    "severity": "MEDIUM",
                    "detail": f"{description}: {value}",
                    "message": "Value NOT found in paper — AI assumed this"
                })

    
    # Check optimizer
    if "optim.Adam" in code and "Adam" not in paper_text:
        assumptions.append({
            "type": "ASSUMED_OPTIMIZER",
            "severity": "MEDIUM", 
            "detail": "Optimizer: Adam",
            "message": "Adam optimizer not mentioned in paper — AI assumed it"
        })

    # Calculate hallucination score
    total_decisions = len(assumptions) + len(from_paper) + len(flags)
    if total_decisions == 0:
        hallucination_score = 100
    else:
        assumption_weight = len(assumptions) * 1 + len(flags) * 2
        hallucination_score = max(0, round(100 - (assumption_weight / total_decisions * 50), 1))

    return {
        "flags": flags,
        "assumptions": assumptions,
        "from_paper": from_paper,
        "hallucination_score": hallucination_score,
        "total_assumptions": len(assumptions) + len(flags),
        "total_from_paper": len(from_paper),
        "summary": f"{len(from_paper)} values from paper, {len(assumptions) + len(flags)} assumptions made"
    }

def format_hallucination_report(analysis):
    """Format hallucination analysis for display"""
    
    report = "\n--- HALLUCINATION ANALYSIS ---\n"
    report += f"🧠 Hallucination Score: {analysis['hallucination_score']}/100\n"
    report += f"📄 From Paper: {analysis['total_from_paper']} values\n"
    report += f"⚠️  Assumptions: {analysis['total_assumptions']} values\n"
    report += f"📊 Summary: {analysis['summary']}\n"

    if analysis['flags']:
        report += "\n🚨 HIGH SEVERITY FLAGS:\n"
        for f in analysis['flags']:
            report += f"  - {f['detail']}: {f['message']}\n"

    if analysis['assumptions']:
        report += "\n⚠️  AI ASSUMPTIONS (not in paper):\n"
        for a in analysis['assumptions']:
            report += f"  - {a['detail']}: {a['message']}\n"

    if analysis['from_paper']:
        report += "\n✅ VALUES FROM PAPER:\n"
        for p in analysis['from_paper']:
            report += f"  - {p['detail']}\n"

    return report