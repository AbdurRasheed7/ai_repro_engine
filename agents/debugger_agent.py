import ollama
from config import MODEL_NAME
import subprocess
import sys

FORCED_IMPORTS = """import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed(42)
np.random.seed(42)

"""

def force_imports(code):
    """Always ensure correct imports are present"""
    lines = code.split("\n")
    cleaned = [l for l in lines if not l.strip().startswith("import torch")
               and not l.strip().startswith("import numpy")
               and not l.strip().startswith("from torch")
               and not l.strip().startswith("from torchvision")
               and not l.strip().startswith("torch.manual_seed")
               and not l.strip().startswith("np.random.seed")]
    return FORCED_IMPORTS + "\n".join(cleaned)

MAX_RETRIES = 3

def extract_error(stderr):
    """Extract the most relevant error from stderr"""
    if not stderr:
        return None
    lines = stderr.strip().split('\n')
    error_lines = [l for l in lines if 'Error' in l or 'error' in l or 'Exception' in l]
    if error_lines:
        return '\n'.join(error_lines[-3:])
    return '\n'.join(lines[-5:])

def auto_install_dependencies(stderr):
    """Detect and install missing packages automatically"""
    if not stderr:
        return False
    
    import re
    
    # Find missing module name
    match = re.search(r"No module named '([^']+)'", stderr)
    if not match:
        return False
    
    package = match.group(1)
    
    # Map module names to pip package names
    package_map = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4',
        'nltk': 'nltk',
        'spacy': 'spacy',
        'transformers': 'transformers',
        'datasets': 'datasets',
    }
    
    pip_name = package_map.get(package, package)
    
    print(f"📦 Auto-installing missing package: {pip_name}...")
    
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "-q"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ Successfully installed {pip_name}!")
        return True
    else:
        print(f"❌ Could not install {pip_name}")
        return False

def fix_code(code, error, attempt):
    """Ask Ollama to fix the broken code"""
    print(f"🔧 Debugger Agent fixing code (attempt {attempt}/{MAX_RETRIES})...")
    
    prompt = f"""You are an expert Python debugger. Fix the error in this code.

STEP 1 - Understand the error:
{error}

STEP 2 - Environment context:
- Framework: PyTorch
- Dataset: MNIST (28x28, 1 channel, 10 classes)
- Input shape: [batch_size, 1, 28, 28]
- Output classes: 10
- Common flatten sizes: 32*13*13=5408, 64*5*5=1600, 32*6*6=1152
- Python version: 3.x
- Available packages: torch, torchvision, numpy, sklearn, matplotlib

STEP 3 - Common error fixes:
- "out of bounds" or "Target X" → wrong number of output classes, use 10
- "shape invalid" or "size mismatch" → wrong flatten size, recalculate
- "not defined" → missing import or variable, add it
- "ModuleNotFoundError" → package missing, use alternative from available packages
- "dimension" or "expected" → wrong tensor shape, fix reshape/view
- "index" errors → wrong indexing, fix array access
- "attribute" errors → wrong method name, use correct PyTorch API
- "value" errors → wrong parameter type or range, fix the value
- "runtime" errors → logic error in forward pass, fix the calculation

STEP 4 - Rules:
- Fix ONLY what the error says, nothing else
- Never remove existing imports
- Always keep torch.manual_seed(42) and np.random.seed(42)
- Always keep print(f"Final Accuracy: {{accuracy:.2f}}%") at end
- If module not found, remove it and use standard alternative
- Return complete working Python code only, no explanation

BROKEN CODE:
{code}

FIXED CODE:"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    
    fixed_code = response['message']['content']
    
    if "```python" in fixed_code:
        fixed_code = fixed_code.split("```python")[1].split("```")[0]
    elif "```" in fixed_code:
        fixed_code = fixed_code.split("```")[1].split("```")[0]
    
    return force_imports(fixed_code)

def run_with_debug(code, code_path):
    """Run code and auto fix if it fails"""
    
    attempt = 0
    current_code = code
    
    while attempt < MAX_RETRIES:
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(current_code)
        
        result = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        has_error = bool(stderr and ("Error" in stderr or "Exception" in stderr))
        has_output = bool(stdout and "Final Accuracy" in stdout)
        
        if has_output and not has_error:
            print(f"✅ Code ran successfully on attempt {attempt + 1}!")
            return stdout, stderr, current_code, attempt + 1
        
        if has_error:
            error = extract_error(stderr)
            print(f"❌ Error detected: {error[:100]}...")
            
            # Try auto-installing missing packages first
            if "No module named" in stderr:
                installed = auto_install_dependencies(stderr)
                if installed:
                    # Retry without changing code
                    attempt += 1
                    continue            
            if attempt < MAX_RETRIES - 1:
                current_code = fix_code(current_code, error, attempt + 1)
            else:
                print("❌ Max retries reached — could not fix code!")
                return stdout, stderr, current_code, attempt + 1
        
        attempt += 1
    
    return stdout, stderr, current_code, attempt