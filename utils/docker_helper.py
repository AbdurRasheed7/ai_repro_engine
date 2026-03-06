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

load_dotenv()

# LLM instance (same as other agents)
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

    prompt = f"""You are a precise DevOps expert. Respond with **EXACTLY** these two code blocks and **ABSOLUTELY NOTHING ELSE** — no introductions, no explanations, no extra text, no markdown headers outside the blocks, no asterisks, no "text" or comments, no bold, no trailing words.

Given this Python code:
{code}

Output ONLY:

**requirements.txt**'''
torch
torchvision
numpy
add any other packages needed from the code
text**Dockerfile**
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
USER nobody
CMD ["python", "solution.py"]
textDo not add any text before, after, or inside the blocks. Do not use * or bold. Do not explain. Just the two blocks above with correct content."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        print("Raw Groq response (for debugging):")
        print(content)

        # Super robust parsing: ignore any header lines inside blocks
        blocks = content.split("```")
        if len(blocks) < 3:
            raise ValueError("Not enough code blocks in response")

        # First block = requirements.txt (skip any header line like "requirements.txt")
        reqs_block = blocks[1].strip()
        reqs_lines = reqs_block.splitlines()
        reqs = '\n'.join([line.strip() for line in reqs_lines if line.strip() and not line.lower().startswith('requirements.txt') and not line.startswith('**') and not line.startswith('*')])

        # Second block = Dockerfile (skip "Dockerfile" line)
        docker_block = blocks[3].strip() if len(blocks) > 3 else blocks[1].strip()
        docker_lines = docker_block.splitlines()
        dockerfile = '\n'.join([line.strip() for line in docker_lines if line.strip() and not line.lower().startswith('dockerfile') and not line.startswith('**') and not line.startswith('*')])

        # Fallback if empty
        if not reqs:
            reqs = "torch\ntorchvision\nnumpy"
        if not dockerfile or 'FROM' not in dockerfile:
            dockerfile = """FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["python", "solution.py"]"""

        print("   Parsed successfully!")
        return {"requirements": reqs, "dockerfile": dockerfile}

    except Exception as e:
        print(f"❌ Failed to generate/parse Docker files: {e}")
        # Strong fallback - always works
        return {
            "requirements": "torch\ntorchvision\nnumpy",
            "dockerfile": """FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER nobody

CMD ["python", "solution.py"]"""
        }


def run_code_in_docker(code: str, paper_id: str, timeout: int = 900) -> tuple[bool, str, str]:
    """
    Build Docker image from generated code + Groq files, run it in container,
    return (success: bool, logs: str, error: str)
    """
    print(f"🐳 Starting Docker execution for paper {paper_id}...")

    client = docker.from_env()  # requires Docker Desktop/daemon running

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save the generated code
        code_path = os.path.join(tmpdir, "solution.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 2. Generate & save Docker files
        docker_files = generate_docker_files(code, paper_id)

        reqs_path = os.path.join(tmpdir, "requirements.txt")
        docker_path = os.path.join(tmpdir, "Dockerfile")

        with open(reqs_path, "w") as f:
            f.write(docker_files["requirements"])

        with open(docker_path, "w") as f:
            f.write(docker_files["dockerfile"])

        # 3. Build the image
        print("   Building Docker image...")
        try:
            image, build_logs = client.images.build(
                path=tmpdir,
                tag=f"repro-engine-{paper_id.lower()}",
                rm=True,
                forcerm=True,
                quiet=False
            )
            print("   Image built successfully")
        except Exception as e:
            error_msg = f"Docker build failed: {str(e)}"
            print(f"❌ {error_msg}")
            return False, "", error_msg

        # 4. Run the container
        print("   Starting container...")
        try:
            container = client.containers.run(
                image.id,
                command="python solution.py",
                detach=True,
                network_mode="none",           # block internet access
                mem_limit="4g",
                memswap_limit="4g",
                remove=True,                   # auto-remove after exit
                stdout=True,
                stderr=True,
                # user="nobody",               # uncomment if base image supports it
            )

            # Stream logs live
            logs = ""
            for chunk in container.logs(stream=True):
                decoded = chunk.decode(errors="ignore")
                logs += decoded
                print(decoded, end="")  # live output in terminal

            # Wait for completion
            result = container.wait(timeout=timeout)
            success = result["StatusCode"] == 0

            print(f"\n   Container finished. Success: {success}")
            return success, logs, "" if success else logs

        except docker.errors.ContainerError as e:
            error_msg = f"Container error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Unexpected Docker error: {str(e)}"
            print(f"❌ {error_msg}")
            return False, "", error_msg