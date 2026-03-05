import ollama
import re
from config import MODEL_NAME, RANDOM_SEED

def generate_code(filtered_text, domain="ml"):
    print("🤖 Sending to Ollama for code generation...")

    if domain == "ml":
        prompt = f"""You are an expert ML engineer. Write a complete runnable Python script.

STRICT RULES:
- Use MNIST dataset from torchvision.datasets.MNIST
- MNIST images are 28x28 pixels with 1 channel (grayscale)
- MNIST has exactly 10 output classes (digits 0-9)
- Last layer must always be nn.Linear(X, 10)
- Use PyTorch only, no Keras or TensorFlow
- Use small model: Conv2d max 32 filters first layer, 64 second layer
- Keep model lightweight for CPU training
- Always use torch.manual_seed(42) and np.random.seed(42)
- Use transforms.Normalize((0.1307,), (0.3081,)) for MNIST
- Train for exactly 5 epochs only
- Use batch_size of 128 for faster training
- Always print: print(f"Final Accuracy: {{accuracy:.2f}}%") at the end
- If anything is unclear write # UNKNOWN - assumed default

PAPER CONTEXT:
{filtered_text}

Write the complete Python script now:"""

    elif domain == "nlp":
        prompt = f"""You are an expert NLP engineer. Write a complete runnable Python script.

STRICT RULES:
- Use torchtext or sklearn for text classification
- Use a simple dataset like 20newsgroups from sklearn
- from sklearn.datasets import fetch_20newsgroups
- Use TF-IDF vectorization + simple classifier
- Use torch.manual_seed(42) and np.random.seed(42)
- Train a simple text classification model
- Always print: print(f"Final Accuracy: {{accuracy:.2f}}%") at the end
- If anything is unclear write # UNKNOWN - assumed default

PAPER CONTEXT:
{filtered_text}

Write the complete Python script now:"""

    elif domain == "recommendation":
        prompt = f"""You are an expert ML engineer. Write a complete runnable Python script.

STRICT RULES:
- Use MovieLens 100K dataset from ./data/ml-100k/u.data
- Load ratings using pandas: pd.read_csv('./data/ml-100k/u.data', sep='\\t', names=['user_id','item_id','rating','timestamp'])
- Implement Matrix Factorization using PyTorch nn.Embedding
- num_users = 943, num_items = 1682, embedding_dim = 50
- Use torch.manual_seed(42) and np.random.seed(42)
- Split data: 80% train, 20% test
- Train for 5 epochs using Adam optimizer
- Evaluate using RMSE on test set
- Convert to accuracy: accuracy = max(0, 100 - (rmse * 20))
- Always print: print(f"Final Accuracy: {{accuracy:.2f}}%") at the end
- If anything is unclear write # UNKNOWN - assumed default

PAPER CONTEXT:
{filtered_text}

Write the complete Python script now:"""

    elif domain == "rl":
        prompt = f"""You are an expert RL engineer. Write a complete runnable Python script.

STRICT RULES:
- Use OpenAI Gym CartPole-v1 environment
- import gym
- Implement a simple DQN or Policy Gradient agent
- Use torch.manual_seed(42) and np.random.seed(42)
- Train for 200 episodes maximum
- Evaluate average reward over last 10 episodes
- Convert reward to accuracy: accuracy = min(100, avg_reward / 2)
- Always print: print(f"Final Accuracy: {{accuracy:.2f}}%") at the end
- If anything is unclear write # UNKNOWN - assumed default

PAPER CONTEXT:
{filtered_text}

Write the complete Python script now:"""

    elif domain == "graph":
            prompt = f"""You are an expert ML engineer. Write a complete runnable Python script.

STRICT RULES:
- Use PyTorch Geometric or simple synthetic graph data
- Create synthetic graph: 100 nodes, random edges
- Implement a simple 2-layer Graph Neural Network (GNN)
- Use torch.manual_seed(42) and np.random.seed(42)
- Node features: random 16-dim vectors
- Binary node classification task
- Train for 5 epochs
- Always print: print(f"Final Accuracy: {{accuracy:.2f}}%") at the end
- If anything is unclear write # UNKNOWN - assumed default

PAPER CONTEXT:
{filtered_text}

Write the complete Python script now:"""

    elif domain == "algorithm":
        prompt = f"""You are an expert software engineer. Write a complete runnable Python script.

STRICT RULES:
- Implement the algorithm described in the context below
- Use only standard Python libraries (no ML frameworks needed)
- Add clear comments explaining each step
- Include test cases with print statements showing results
- Always print: print(f"Algorithm Result: {{result}}")
- If anything is unclear write # UNKNOWN - assumed default

ALGORITHM CONTEXT:
{filtered_text}

Write the complete Python implementation now:"""

    else:
        prompt = f"""You are an expert ML engineer. Write a complete runnable Python script.

STRICT RULES:
- Use MNIST dataset from torchvision.datasets.MNIST
- MNIST has exactly 10 output classes (digits 0-9)
- Use PyTorch only
- Use torch.manual_seed(42) and np.random.seed(42)
- Train for exactly 5 epochs only
- Always print: print(f"Final Accuracy: {{accuracy:.2f}}%") at the end

PAPER CONTEXT:
{filtered_text}

Write the complete Python script now:"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    code = response['message']['content']

    # Clean up markdown code blocks if present
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    if domain in ["ml", "nlp", "recommendation", "rl"]:
        # Remove existing import lines to avoid duplicates
        lines = code.split("\n")
        cleaned_lines = [l for l in lines if not l.strip().startswith("import torch")
                        and not l.strip().startswith("import numpy")
                        and not l.strip().startswith("from torch")
                        and not l.strip().startswith("from torchvision")
                        and not l.strip().startswith("torch.manual_seed")
                        and not l.strip().startswith("np.random.seed")]
        code = "\n".join(cleaned_lines)

        # Force all required imports at top
        forced_imports = f"""import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

"""
        code = forced_imports + code

        # Fix common output class errors
        code = re.sub(r'nn\.Linear\(([^,]+),\s*[1-9]\)',
                     lambda m: f'nn.Linear({m.group(1)}, 10)',
                     code)

    print("✅ Code generated successfully!")
    return code