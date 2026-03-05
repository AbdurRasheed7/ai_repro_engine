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

# Known assumption patterns in code (expanded)
ASSUMPTION_PATTERNS = [
    (r"learning_rate\s*=\s*[\d.eE+-]+", "Learning rate value"),
    (r"lr\s*=\s*[\d.eE+-]+", "Learning rate (short form)"),
    (r"batch_size\s*=\s*[\d]+", "Batch size value"),
    (r"num_epochs\s*=\s*[\d]+", "Number of epochs"),
    (r"epochs\s*=\s*[\d]+", "Epochs (short form)"),
    (r"hidden_size\s*=\s*[\d]+", "Hidden layer size"),
    (r"dropout\s*=\s*[\d.]+", "Dropout rate"),
    (r"momentum\s*=\s*[\d.]+", "Momentum value"),
    (r"weight_decay\s*=\s*[\d.eE+-]+", "Weight decay value"),
    (r"nn\.Linear\([\d,\s]+\)", "Fully connected layer size"),
    (r"nn\.Conv2d\([\d,\s]+\)", "Convolution layer parameters"),
    (r"kernel_size\s*=\s*[\d]+", "Kernel size"),
    (r"padding\s*=\s*[\d]+", "Padding value"),
    (r"stride\s*=\s*[\d]+", "Stride value"),
    (r"optim\.Adam", "Optimizer type (Adam assumed)"),
    (r"optim\.SGD", "Optimizer type (SGD assumed)"),
    (r"optim\.RMSprop", "Optimizer type (RMSprop assumed)"),
    (r"nn\.(ReLU|LeakyReLU|Sigmoid|Tanh|Softmax)", "Activation function"),
    (r"criterion\s*=\s*nn\.(CrossEntropyLoss|MSELoss|BCELoss)", "Loss function"),
]

def analyze_hallucinations(code, paper_text):
    """Analyze generated code for hallucinations and assumptions"""
    
    flags = []
    assumptions = []
    from_paper = []

    text_lower = paper_text.lower()

    # 1. Explicit UNKNOWN flags
    unknown_lines = [l.strip() for l in code.split('\n') if 'UNKNOWN' in l]
    for line in unknown_lines:
        flags.append({
            "type": "EXPLICIT_ASSUMPTION",
            "severity": "HIGH",
            "detail": line,
            "message": "AI explicitly flagged this as unknown"
        })

    # 2. Check each pattern
    for pattern, description in ASSUMPTION_PATTERNS:
        matches = re.findall(pattern, code)
        for match in matches:
            value_str = match if isinstance(match, str) else match[0]
            
            # Extract numeric part if present
            numbers = re.findall(r'[\d.eE+-]+', value_str)
            found_in_paper = False
            
            # Better matching: look for context around number in paper
            for num in numbers:
                # Check if number appears near hyperparam words in paper
                context_pattern = rf"(lr|learning rate|batch size|epochs?|hidden|dropout|momentum|weight decay|kernel|stride|padding)\s*[:=]\s*{re.escape(num)}"
                if re.search(context_pattern, paper_text, re.IGNORECASE):
                    found_in_paper = True
                    break
                # Loose check: number in paper + nearby hyperparam word
                if num in paper_text and any(word in paper_text for word in ["lr", "batch", "epoch", "hidden", "dropout"]):
                    found_in_paper = True
                    break
            
            if found_in_paper:
                from_paper.append({
                    "detail": f"{description}: {value_str}",
                    "message": "Value likely from paper ✅"
                })
            else:
                severity = "HIGH" if "optimizer" in description.lower() else "MEDIUM"
                assumptions.append({
                    "type": "ASSUMED_VALUE",
                    "severity": severity,
                    "detail": f"{description}: {value_str}",
                    "message": "Value NOT clearly in paper — AI assumed"
                })

    # 3. Reverse optimizer check (paper specifies different optimizer)
    code_optimizers = re.findall(r"optim\.(Adam|SGD|RMSprop|AdamW)", code)
    paper_optimizers = re.findall(r"(Adam|SGD|RMSprop|AdamW)", paper_text, re.IGNORECASE)
    for opt in code_optimizers:
        if opt.lower() not in [o.lower() for o in paper_optimizers]:
            assumptions.append({
                "type": "ASSUMED_OPTIMIZER",
                "severity": "HIGH",
                "detail": f"Optimizer: {opt}",
                "message": "Optimizer in code not mentioned in paper"
            })

    # 4. Calculate improved score (weighted by severity)
    total_decisions = len(assumptions) + len(from_paper) + len(flags)
    if total_decisions == 0:
        hallucination_score = 100
    else:
        # Weight: HIGH=3, MEDIUM=1
        weighted_assumptions = sum(3 if a['severity'] == 'HIGH' else 1 for a in assumptions) + len(flags) * 3
        hallucination_score = max(0, round(100 - (weighted_assumptions / total_decisions * 60), 1))  # 60 max penalty

    return {
        "flags": flags,
        "assumptions": assumptions,
        "from_paper": from_paper,
        "hallucination_score": hallucination_score,
        "total_assumptions": len(assumptions) + len(flags),
        "total_from_paper": len(from_paper),
        "summary": f"{len(from_paper)} values from paper, {len(assumptions) + len(flags)} assumptions made (weighted score)"
    }

def format_hallucination_report(analysis):
    """Format hallucination analysis for display"""
    
    report = "\n--- HALLUCINATION ANALYSIS ---\n"
    report += f"🧠 Hallucination Score: {analysis['hallucination_score']}/100 (higher = better)\n"
    report += f"📄 From Paper: {analysis['total_from_paper']} values\n"
    report += f"⚠️  Assumptions: {analysis['total_assumptions']} values\n"
    report += f"📊 Summary: {analysis['summary']}\n"

    if analysis['flags']:
        report += "\n🚨 HIGH SEVERITY FLAGS:\n"
        for f in analysis['flags']:
            report += f"  - {f['detail']}: {f['message']}\n"

    if analysis['assumptions']:
        report += "\n⚠️  AI ASSUMPTIONS (not clearly in paper):\n"
        for a in analysis['assumptions']:
            report += f"  - {a['detail']} ({a['severity']}): {a['message']}\n"

    if analysis['from_paper']:
        report += "\n✅ VALUES FROM PAPER:\n"
        for p in analysis['from_paper']:
            report += f"  - {p['detail']}\n"

    return report