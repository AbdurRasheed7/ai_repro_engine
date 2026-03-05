"""
utils/docker_helper.py

Helper functions for generating and running Docker containers for reproducibility.
Uses Groq to auto-generate Dockerfile + requirements.txt from generated code.
"""

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import docker
import tempfile
import shutil

load_dotenv()

# Reuse the same LLM instance as other agents (or define here)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=2048,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def generate_docker_files(code: str, paper_id: str = "temp") -> dict:
    """
    Ask Groq to generate minimal requirements.txt and Dockerfile based on the code.
    Returns dict with 'requirements' and 'dockerfile' strings.
    """
    print(f"🐳 Generating Docker files for paper {paper_id}...")

    prompt = f"""You are an expert DevOps engineer specializing in reproducible ML environments.

Given this Python script:
{code}

Generate **exactly two files** needed to run this code in Docker:

1. **requirements.txt**  
   - List EVERY package imported or used in the code (exact versions if obvious, otherwise latest stable).  
   - Include torch, torchvision if used.  
   - Include numpy, etc.  
   - Do NOT include unnecessary tools like docker, jupyter, etc.  
   - One package per line.

2. **Dockerfile**  
   - Minimal, secure, CPU-based (no GPU needed for testing).  
   - Use FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime (or CPU version if no CUDA).  
   - WORKDIR /app  
   - COPY requirements.txt .  
   - RUN pip install --no-cache-dir -r requirements.txt  
   - COPY . .  
   - USER nobody  # security best practice  
   - CMD ["python", "solution.py"]  
   - Add HEALTHCHECK or timeout if possible.

Output **ONLY** in this exact format (no explanations, no extra text):

**requirements.txt**"""