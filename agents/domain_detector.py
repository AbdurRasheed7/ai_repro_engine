import re

# Domain keywords (added more CV-specific terms)
DOMAIN_KEYWORDS = {
    "image_classification": [
        "image classification", "convolutional", "CNN", "conv2d", "image recognition",
        "MNIST", "CIFAR", "ImageNet", "pixel", "visual", "object detection",
        "feature maps", "pooling", "batch normalization", "image dataset",
        "visualizing activations", "saliency maps", "occlusion", "layer visualization",
        "convolutional neural network", "residual block", "resnet", "vgg", "alexnet"
    ],
    "nlp": [ ... ]  # unchanged, keep your list
    # ... other domains unchanged
}

# Dataset mapping unchanged
DOMAIN_DATASETS = {
    "image_classification": "MNIST (28x28 grayscale, 10 classes)",
    "nlp": "20NewsGroups (text classification)",
    "recommendation": "MovieLens 100K (user-item ratings)",
    "reinforcement_learning": "CartPole-v1 (OpenAI Gym)",
    "algorithm": "Custom test cases (pure Python)",
    "generative": "MNIST (28x28 grayscale, for generation)",
    "graph": "Synthetic graph data (node classification)",
    "unknown": "Unable to determine — manual selection needed"
}  # your original

def detect_domain(paper_text):
    text_lower = paper_text.lower()

    # ── Hard Override Rules ────────────────────────────────

    # Graph / 3D — first (most specific) — unchanged

    # Add early CV boost (before other overrides)
    if any(term in text_lower for term in ["convolutional", "convnet", "cnn", "resnet", "vgg"]):
        return {
            "domain": "image_classification",
            "confidence": 95,
            "score": 99,
            "matched_keywords": ["convolutional / CNN / classic arch detected"],
            "dataset": DOMAIN_DATASETS["image_classification"],
            "all_scores": {}
        }

    # Recommendation — unchanged
    # NLP — unchanged
    # RL — unchanged
    # Generative — unchanged
    # Algorithm — unchanged

    # Image Classification fallback (now stronger due to keywords)
    if any(term in text_lower for term in DOMAIN_KEYWORDS["image_classification"]):
        return {
            "domain": "image_classification",
            "confidence": 90,
            "score": 99,
            "matched_keywords": ["image classification detected"],
            "dataset": DOMAIN_DATASETS["image_classification"],
            "all_scores": {}
        }

    # ── Fallback scoring ─────────────────────────
    scores = {}
    matched_keywords = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        matches = []
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            if count > 0:
                score += count
                matches.append(keyword)
        scores[domain] = score
        matched_keywords[domain] = matches

    # Extra boost for CV terms in fallback
    if "conv" in text_lower or "convolution" in text_lower:
        scores["image_classification"] += 5

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    if best_score < 2:
        best_domain = "unknown"

    sorted_scores = sorted(scores.values(), reverse=True)
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0

    if best_score == 0:
        confidence = 0
    elif second_score == 0:
        confidence = 100
    else:
        confidence = min(100, int((best_score / (best_score + second_score)) * 100))

    if confidence < 60:
        best_domain = "unknown"

    return {
        "domain": best_domain,
        "confidence": confidence,
        "score": best_score,
        "matched_keywords": matched_keywords.get(best_domain, [])[:5],
        "dataset": DOMAIN_DATASETS.get(best_domain, "Unknown"),
        "all_scores": scores
    }

# format_domain_report (small improvement)
def format_domain_report(detection):
    domain = detection['domain']
    confidence = detection['confidence']
    keywords = detection['matched_keywords'][:5]

    report = "\n--- DOMAIN DETECTION ---\n"

    if domain == "unknown":
        report += "⚠️  Domain: UNKNOWN — Could not determine paper domain\n"
        report += "💡 Defaulting to image classification (project focus)\n"
    else:
        emoji_map = {
            "image_classification": "🖼️",
            "nlp": "📝",
            "recommendation": "🎯",
            "reinforcement_learning": "🤖",
            "algorithm": "⚙️",
            "graph": "🕸️",
            "generative": "🎨"
        }
        emoji = emoji_map.get(domain, "📄")
        report += f"{emoji}  Domain: {domain.replace('_', ' ').upper()}\n"
        report += f"📊 Confidence: {confidence}%\n"
        report += f"🔑 Key terms found: {', '.join(keywords)}\n"
        report += f"💾 Will use dataset: {detection['dataset']}\n"

    return report

# get_code_domain — unchanged
def get_code_domain(detection):
    domain = detection['domain']

    mapping = {
        "image_classification": "ml",
        "generative": "generative",
        "nlp": "nlp",
        "recommendation": "recommendation",
        "reinforcement_learning": "rl",
        "algorithm": "algorithm",
        "graph": "graph",
        "unknown": "ml"
    }

    return mapping.get(domain, "ml")