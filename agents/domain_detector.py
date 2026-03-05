import re

# Domain keywords
DOMAIN_KEYWORDS = {
    "image_classification": [
        "image classification", "convolutional", "CNN", "conv2d", "image recognition",
        "MNIST", "CIFAR", "ImageNet", "pixel", "visual", "object detection",
        "feature maps", "pooling", "batch normalization", "image dataset"
    ],
    "nlp": [
        "natural language", "text classification", "sentiment analysis", "BERT",
        "transformer", "word embedding", "tokenization", "language model",
        "sequence to sequence", "named entity", "text dataset", "corpus",
        "vocabulary", "NLP", "sentence", "word2vec", "GloVe"
    ],
    "recommendation": [
        "recommendation system", "collaborative filtering", "user-item",
        "matrix factorization", "MovieLens", "rating prediction", "user preference",
        "item embedding", "implicit feedback", "explicit feedback", "recommender",
        "user embedding", "item rating", "user behavior", "purchase history",
        "click through", "CTR", "recall@k", "precision@k", "NDCG",
        "cold start", "sparse matrix", "interaction matrix", "user profile",
        "item profile", "deep reinforcement learning recommendation",
        "reinforcement learning recommendation", "reward recommendation",
        "policy recommendation", "action recommendation", "state user",
        "e-commerce", "personalized", "top-k recommendation"
    ],
    "reinforcement_learning": [
        "reinforcement learning", "reward function", "policy gradient", "Q-learning",
        "agent", "environment", "action space", "state space", "OpenAI Gym",
        "Markov decision", "DQN", "PPO", "actor-critic", "episode"
    ],
    "algorithm": [
        "sorting algorithm", "graph algorithm", "dynamic programming",
        "binary search", "tree traversal", "shortest path", "complexity analysis",
        "data structure", "hash table", "linked list", "recursion"
    ],
    "graph": [
    "graph neural network", "point cloud", "graph matching",
    "message passing", "node classification", "graph convolution",
    "edge features", "adjacency matrix", "graph embedding",
    "3d object", "GNN", "GCN", "graph attention network",
    "knowledge graph", "graph network", "graph convolutional",
    "spectral graph", "spatial graph", "graph pooling",
    "node embedding", "link prediction", "graph classification"
    ],
    "generative": [
        "generative adversarial", "GAN", "variational autoencoder", "VAE",
        "image generation", "latent space", "generator", "discriminator",
        "diffusion model", "synthesis"
    ]
}

# Dataset mapping for each domain
DOMAIN_DATASETS = {
    "image_classification": "MNIST (28x28 grayscale, 10 classes)",
    "nlp": "20NewsGroups (text classification)",
    "recommendation": "MovieLens 100K (user-item ratings)",
    "reinforcement_learning": "CartPole-v1 (OpenAI Gym)",
    "algorithm": "Custom test cases (pure Python)",
    "generative": "MNIST (28x28 grayscale, for generation)",
    "graph": "Synthetic graph data (node classification)",
    "unknown": "Unable to determine — manual selection needed"
}

def detect_domain(paper_text):
    """Detect the domain of a research paper"""

    text_lower = paper_text.lower()

    # ── Hard Override Rules ────────────────────────────────

# Graph / 3D Point Cloud — check FIRST (most specific)
    if any(term in text_lower for term in [
        "graph neural network", "point cloud", "graph matching",
        "message passing", "node classification", "graph convolution",
        "edge features", "adjacency matrix", "graph embedding",
        "3d object", "gnn", "gcn", "graph attention",
        "knowledge graph", "graph network"
    ]):
        return {
            "domain": "graph",
            "confidence": 95,
            "score": 99,
            "matched_keywords": ["graph/3D domain detected"],
            "dataset": DOMAIN_DATASETS["graph"],
            "all_scores": {}
        }

    # Recommendation
    if any(term in text_lower for term in [
        "recommendation system", "recommender system",
        "collaborative filtering", "matrix factorization",
        "user-item interaction", "rating prediction",
        "personalized recommendation", "top-k recommendation"
    ]):
        return {
            "domain": "recommendation",
            "confidence": 95,
            "score": 99,
            "matched_keywords": ["recommendation system detected"],
            "dataset": DOMAIN_DATASETS["recommendation"],
            "all_scores": {}
        }

    # NLP
    if any(term in text_lower for term in [
        "sentiment analysis", "text classification",
        "named entity recognition", "machine translation",
        "question answering", "language model", "bert",
        "transformer model", "word embedding", "tokenization",
        "natural language processing", "text generation"
    ]):
        return {
            "domain": "nlp",
            "confidence": 95,
            "score": 99,
            "matched_keywords": ["NLP task detected"],
            "dataset": DOMAIN_DATASETS["nlp"],
            "all_scores": {}
        }

    # Reinforcement Learning
    if any(term in text_lower for term in [
        "reinforcement learning", "reward function",
        "policy gradient", "q-learning", "markov decision",
        "dqn", "ppo", "actor-critic", "openai gym",
        "action space", "state space", "episode reward"
    ]):
        return {
            "domain": "reinforcement_learning",
            "confidence": 95,
            "score": 99,
            "matched_keywords": ["reinforcement learning detected"],
            "dataset": DOMAIN_DATASETS["reinforcement_learning"],
            "all_scores": {}
        }

    # Generative Models
    if any(term in text_lower for term in [
        "generative adversarial", "variational autoencoder",
        "image generation", "gan", "vae", "diffusion model",
        "generator network", "discriminator network",
        "latent space", "image synthesis"
    ]):
        return {
            "domain": "generative",
            "confidence": 95,
            "score": 99,
            "matched_keywords": ["generative model detected"],
            "dataset": DOMAIN_DATASETS["generative"],
            "all_scores": {}
        }

    # Algorithm
    if any(term in text_lower for term in [
        "sorting algorithm", "graph algorithm",
        "dynamic programming", "binary search",
        "tree traversal", "shortest path",
        "time complexity", "space complexity",
        "data structure", "hash table"
    ]):
        return {
            "domain": "algorithm",
            "confidence": 95,
            "score": 99,
            "matched_keywords": ["algorithm detected"],
            "dataset": DOMAIN_DATASETS["algorithm"],
            "all_scores": {}
        }

    # Image Classification
    if any(term in text_lower for term in [
        "image classification", "convolutional neural network",
        "object detection", "image recognition",
        "visual recognition", "cifar", "imagenet"
    ]):
        return {
            "domain": "image_classification",
            "confidence": 90,
            "score": 99,
            "matched_keywords": ["image classification detected"],
            "dataset": DOMAIN_DATASETS["image_classification"],
            "all_scores": {}
        }

    # ── Fallback — keyword scoring ─────────────────────────
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

def format_domain_report(detection):
    """Format domain detection results"""
    domain = detection['domain']
    confidence = detection['confidence']
    keywords = detection['matched_keywords'][:5]

    report = "\n--- DOMAIN DETECTION ---\n"

    if domain == "unknown":
        report += "⚠️  Domain: UNKNOWN — Could not determine paper domain\n"
        report += "💡 Defaulting to image classification\n"
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

def get_code_domain(detection):
    """Map detected domain to coder agent domain parameter"""
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