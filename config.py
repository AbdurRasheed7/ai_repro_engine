import os

# Paths
BASE_DIR = r"D:\ai_repro_engine"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_code")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

# Ollama Model
MODEL_NAME = "qwen2.5-coder:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reproducibility
RANDOM_SEED = 42

# Section filter keywords
KEEP_KEYWORDS = [
    "method", "architecture", "model", "approach",
    "training", "experiment", "implement",
    "hyperparameter", "optimizer", "dataset",
    "result", "evaluat", "benchmark", "accuracy",
    "loss", "batch", "epoch", "layer", "network"
]

STOP_SECTIONS = [
    "conclusion", "references", "appendix",
    "acknowledgment", "future work", "related work",
    "supplementary", "supplemental"
]