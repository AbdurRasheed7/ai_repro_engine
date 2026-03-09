import os

# ── Project Paths ───────────────────────────────────────────────
BASE_DIR   = r"D:\ai_repro_engine"
INPUT_DIR  = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_code")
REPORTS_DIR= os.path.join(BASE_DIR, "reports")
PROMPTS_DIR= os.path.join(BASE_DIR, "prompts")
TESTS_DIR  = os.path.join(BASE_DIR, "tests")
GOLDEN_DIR = os.path.join(TESTS_DIR, "golden")

# ── LLM Configuration (Groq) ────────────────────────────────────
# Used by: coder_agent, debugger_agent, golden_agent, docker_helper
# Agents that need different settings override locally
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.1
GROQ_MAX_TOKENS  = 8192

# ── RAG Settings ────────────────────────────────────────────────
CHUNK_SIZE      = 800   # chars per chunk
CHUNK_OVERLAP   = 150   # overlap between chunks to avoid cutting sentences
TOP_K_CHUNKS    = 6     # chunks retrieved per query
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
    "proof", "theorem", "bibliography"
]

# ── Docker / Execution Settings ─────────────────────────────────
# NOTE: docker_helper.py uses python:3.10-slim-bookworm (CPU-only, lean image)
# CODE_TIMEOUT_SEC and MAX_MEMORY_MB are passed to docker_helper.run_code_in_docker
CODE_TIMEOUT_SEC = 900    # 15 minutes max container run time
MAX_MEMORY_MB    = 4096   # container memory limit (4GB)

# ── Report & UI Settings ────────────────────────────────────────
REPORT_TEMPLATE    = "default"
EXPECTED_ACC_DEFAULT = 95.0
TOLERANCE_DEFAULT    = 2.0