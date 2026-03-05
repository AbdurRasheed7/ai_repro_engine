import os

# ── Project Paths ───────────────────────────────────────────────
BASE_DIR = r"D:\ai_repro_engine"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_code")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
GOLDEN_DIR = os.path.join(TESTS_DIR, "golden")

# ── LLM Configuration (Groq) ────────────────────────────────────
# No MODEL_NAME needed anymore — handled in each agent file
# Groq API key is loaded from .env (via dotenv)
GROQ_MODEL = "llama-3.3-70b-versatile"  # default, can override per agent
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS = 8192

# ── RAG Settings ────────────────────────────────────────────────
CHUNK_SIZE = 800          # increased a bit for better context
CHUNK_OVERLAP = 150       # more overlap to avoid cutting sentences
TOP_K_CHUNKS = 6          # retrieve a few more for richer context
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Reproducibility & Randomness ────────────────────────────────
RANDOM_SEED = 42

# ── Section Filtering for Parser ────────────────────────────────
KEEP_KEYWORDS = [
    "method", "architecture", "model", "approach",
    "training", "experiment", "implement", "implementation",
    "hyperparameter", "optimizer", "dataset", "preprocess",
    "result", "evaluat", "benchmark", "accuracy", "top-1",
    "loss", "batch", "epoch", "layer", "network", "convolution",
    "residual", "attention", "transformer", "head", "block"
]

STOP_SECTIONS = [
    "conclusion", "conclusions", "discussion", "references",
    "appendix", "acknowledgment", "acknowledgements", "future work",
    "related work", "supplementary", "supplemental", "ablation",
    "proof", "theorem", "appendix", "bibliography"
]

# ── Docker / Execution Settings ─────────────────────────────────
CODE_TIMEOUT_SEC = 900          # 15 minutes max run time
DOCKER_BASE_IMAGE = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"  # good for CV
MAX_MEMORY_MB = 4096            # limit container memory

# ── Report & UI Settings ────────────────────────────────────────
REPORT_TEMPLATE = "default"     # can add more later
EXPECTED_ACC_DEFAULT = 95.0
TOLERANCE_DEFAULT = 2.0