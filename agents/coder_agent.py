import re
from config import RANDOM_SEED
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Groq LLM once (outside the function for efficiency)
from config import RANDOM_SEED, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=GROQ_TEMPERATURE,
    max_tokens=GROQ_MAX_TOKENS,
)


# ── Proven working templates for each domain ─────────────────────────────────
# These are injected into generated code via post-processing.
# Groq only fills in hyperparameters — the architecture is always correct.

ML_TEMPLATE = f"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# ── Proven LeNet-style model (97-99% on MNIST in 5 epochs) ───────────────────
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.LazyLinear(128)
        self.fc2   = nn.Linear(128, 10)
        self.drop  = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)   # dynamic flatten — never hardcoded
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# ── Data loading ──────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=128, shuffle=False)

# ── Training ──────────────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{{epoch+1}}/{{num_epochs}}] Loss: {{running_loss/len(train_loader):.4f}}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

NLP_TEMPLATE = f"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading 20newsgroups dataset...")
data   = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), data_home='/app/data')
texts  = data.data
labels = data.target

# ── TF-IDF vectorization ──────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
y = np.array(labels, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state={RANDOM_SEED})

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
num_classes  = len(np.unique(y))
input_dim    = X_train.shape[1]

class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        self.fc1  = nn.Linear(input_dim, 256)
        self.bn1  = nn.BatchNorm1d(256)
        self.fc2  = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

import torch.nn.functional as F

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = TextClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── Training ──────────────────────────────────────────────────────────────────
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{{epoch+1}}/{{num_epochs}}] Loss: {{running_loss/len(train_loader):.4f}}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
correct = 0
total   = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs  = model(X_batch)
        _, preds = torch.max(outputs, 1)
        total   += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

accuracy = 100.0 * correct / total
print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

RECOMMENDATION_TEMPLATE = f"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading MovieLens 100K dataset...")
ratings = pd.read_csv('./data/ml-100k/u.data', sep='\\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

# Dynamic sizing — never hardcode 943 or 1682
num_users = ratings['user_id'].max() + 1
num_items = ratings['item_id'].max() + 1
print(f"Users: {{num_users}}, Items: {{num_items}}")

user_ids = torch.tensor(ratings['user_id'].values, dtype=torch.long)
item_ids = torch.tensor(ratings['item_id'].values, dtype=torch.long)
rating_vals = torch.tensor(ratings['rating'].values, dtype=torch.float32)

# ── Train/test split ──────────────────────────────────────────────────────────
indices = list(range(len(ratings)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state={RANDOM_SEED})

train_dataset = TensorDataset(user_ids[train_idx], item_ids[train_idx], rating_vals[train_idx])
test_dataset  = TensorDataset(user_ids[test_idx],  item_ids[test_idx],  rating_vals[test_idx])
train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=256, shuffle=False)

# ── Matrix Factorization model ────────────────────────────────────────────────
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(MatrixFactorization, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.user_bias  = nn.Embedding(num_users, 1)
        self.item_bias  = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        u = self.user_embed(user)
        i = self.item_embed(item)
        ub = self.user_bias(user).squeeze()
        ib = self.item_bias(item).squeeze()
        return (u * i).sum(dim=1) + ub + ib

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = MatrixFactorization(num_users, num_items, embedding_dim=50).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ── Training ──────────────────────────────────────────────────────────────────
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for users, items, rts in train_loader:
        users, items, rts = users.to(device), items.to(device), rts.to(device)
        optimizer.zero_grad()
        preds = model(users, items)
        loss  = criterion(preds, rts)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{{epoch+1}}/{{num_epochs}}] Loss: {{running_loss/len(train_loader):.4f}}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
sq_err = 0.0
count  = 0
with torch.no_grad():
    for users, items, rts in test_loader:
        users, items, rts = users.to(device), items.to(device), rts.to(device)
        preds   = model(users, items)
        sq_err += ((preds - rts) ** 2).sum().item()
        count  += rts.size(0)

rmse     = math.sqrt(sq_err / count)
accuracy = max(0.0, 100.0 - (rmse * 20))
print(f"RMSE: {{rmse:.4f}}")
print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

RL_TEMPLATE = f"""
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})
random.seed({RANDOM_SEED})

# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ── Environment & training ────────────────────────────────────────────────────
env        = gym.make('CartPole-v1')
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model       = DQN(state_dim, action_dim).to(device)
target_net  = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(model.state_dict())
optimizer   = optim.Adam(model.parameters(), lr=1e-3)
criterion   = nn.MSELoss()
buffer      = ReplayBuffer()

epsilon      = 1.0
epsilon_min  = 0.01
epsilon_decay= 0.995
gamma        = 0.99
batch_size   = 64
num_episodes = 500
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset() if hasattr(env.reset(), '__iter__') else (env.reset(), {{}})
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0

    for _ in range(500):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(s).argmax().item()

        result = env.step(action)
        next_state, reward, done = result[0], result[1], result[2]
        buffer.push(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            batch              = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states      = torch.FloatTensor(np.array(states)).to(device)
            actions     = torch.LongTensor(actions).to(device)
            rewards     = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones       = torch.FloatTensor(dones).to(device)

            q_values      = model(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = target_net(next_states).max(1)[0].detach()
            targets       = rewards + gamma * next_q_values * (1 - dones)

            loss = criterion(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"Episode {{episode+1}}/{{num_episodes}} | Avg Reward (last 50): {{avg:.2f}} | Epsilon: {{epsilon:.3f}}")

    if (episode + 1) % 10 == 0:
        target_net.load_state_dict(model.state_dict())

env.close()

# ── Evaluation ────────────────────────────────────────────────────────────────
avg_reward = np.mean(reward_history[-50:])
accuracy   = min(100.0, avg_reward / 5.0)
print(f"Final Accuracy: {{accuracy:.2f}}% (avg reward: {{avg_reward:.2f}})")
"""

GRAPH_TEMPLATE = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# ── Synthetic graph (pure PyTorch — no PyG dependency) ───────────────────────
num_nodes   = 200
num_edges   = 800
num_features= 16
num_classes = 2

# Node features and labels
X = torch.randn(num_nodes, num_features)
y = torch.randint(0, num_classes, (num_nodes,))

# Random adjacency (sparse COO format)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
# Build normalized adjacency matrix
adj = torch.zeros(num_nodes, num_nodes)
adj[edge_index[0], edge_index[1]] = 1.0
adj[edge_index[1], edge_index[0]] = 1.0   # undirected
adj += torch.eye(num_nodes)                # self-loops
deg  = adj.sum(dim=1, keepdim=True).clamp(min=1)
adj  = adj / deg                           # row normalize

# Train/test split
perm       = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:160]] = True
test_mask[perm[160:]]  = True

# ── 2-layer GCN (manual message passing) ─────────────────────────────────────
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = F.relu(self.fc1(adj @ x))
        x = self.drop(x)
        return self.fc2(adj @ x)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = GCN().to(device)
X, y, adj = X.to(device), y.to(device), adj.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# ── Training ──────────────────────────────────────────────────────────────────
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out  = model(X, adj)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {{epoch+1}}/200 | Loss: {{loss.item():.4f}}")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    out      = model(X, adj)
    preds    = out[test_mask].argmax(dim=1)
    correct  = (preds == y[test_mask]).sum().item()
    total    = test_mask.sum().item()
    accuracy = 100.0 * correct / total

print(f"Final Accuracy: {{accuracy:.2f}}%")
"""

ALGORITHM_TEMPLATE = f"""
import numpy as np
import random
random.seed({RANDOM_SEED})
np.random.seed({RANDOM_SEED})

# Algorithm implementation will be inserted by Groq below.
# Template ensures correct imports and seed are always present.
"""


def fix_hardcoded_flatten(code):
    """
    Post-processing fix: if Groq hardcoded a flatten size in nn.Linear,
    replace it with nn.LazyLinear(10) which infers input size automatically.

    Fixes:
    1. x.view(-1, ANYTHING)        → x.view(x.size(0), -1)
    2. x.view(-1,)  shorthand      → x.view(x.size(0), -1)
    3. nn.Linear(4+_DIGIT_NUM, 10) → nn.LazyLinear(10)
    4. nn.Linear(A*B*C, 10)        → nn.LazyLinear(10)
    5. nn.Linear(A*B, 10)          → nn.LazyLinear(10)
    """

    # Fix 1: Replace x.view(-1, anything) → x.view(x.size(0), -1)
    code = re.sub(
        r'x\s*=\s*x\.view\(\s*-1\s*,\s*[^\)]+\)',
        'x = x.view(x.size(0), -1)  # fixed: dynamic flatten',
        code
    )

    # Fix 2: Catch simple shorthand x.view(-1,) edge case
    code = code.replace("x.view(-1,", "x.view(x.size(0), -1)  # fixed")

    # Fix 3: Replace nn.Linear(ANY_4+_DIGIT_NUMBER, 10) → nn.LazyLinear(10)
    code = re.sub(
        r'nn\.Linear\(\s*\d{4,}\s*,\s*10\s*\)',
        'nn.LazyLinear(10)',
        code
    )

    # Fix 4: Catch expressions like nn.Linear(64 * 4 * 4, 10) or nn.Linear(64*7*7, 10)
    code = re.sub(
        r'nn\.Linear\(\s*\d+\s*\*\s*\d+\s*\*\s*\d+\s*,\s*10\s*\)',
        'nn.LazyLinear(10)',
        code
    )

    # Fix 5: Catch expressions like nn.Linear(64 * 4, 10) or nn.Linear(64*4, 10)
    code = re.sub(
        r'nn\.Linear\(\s*\d+\s*\*\s*\d+\s*,\s*10\s*\)',
        'nn.LazyLinear(10)',
        code
    )

    return code


def generate_code(filtered_text, domain="ml"):
    print("🤖 Sending to Groq for code generation...")

    if domain == "ml":
        prompt = f"""You are an expert ML engineer. Extract ONLY the hyperparameters from the paper below.

I already have a working model. I just need you to confirm or adjust these values from the paper:
- learning_rate (default: 0.01)
- momentum (default: 0.9)
- batch_size (default: 128)
- num_epochs (default: 5)
- weight_decay (default: 1e-4)

Reply with ONLY a Python comment block like:
# lr={{}}, momentum={{}}, batch_size={{}}, epochs={{}}, weight_decay={{}}

PAPER CONTEXT:
{filtered_text}"""

    elif domain == "nlp":
        prompt = f"""You are an expert NLP engineer. Extract ONLY the hyperparameters from the paper below.

I already have a working NLP classifier. I just need:
- learning_rate (default: 1e-3)
- batch_size (default: 64)
- num_epochs (default: 10)
- dropout (default: 0.3)

Reply with ONLY a Python comment block like:
# lr={{}}, batch_size={{}}, epochs={{}}, dropout={{}}

PAPER CONTEXT:
{filtered_text}"""

    elif domain == "recommendation":
        prompt = f"""You are an expert recommender systems engineer. Extract ONLY the hyperparameters from the paper below.

I already have a working Matrix Factorization model. I just need:
- learning_rate (default: 1e-3)
- embedding_dim (default: 50)
- num_epochs (default: 5)
- batch_size (default: 256)

Reply with ONLY a Python comment block like:
# lr={{}}, embedding_dim={{}}, epochs={{}}, batch_size={{}}

PAPER CONTEXT:
{filtered_text}"""

    elif domain == "rl":
        prompt = f"""You are an expert RL engineer. Extract ONLY the hyperparameters from the paper below.

I already have a working DQN agent. I just need:
- learning_rate (default: 1e-3)
- gamma (default: 0.99)
- epsilon_decay (default: 0.995)
- num_episodes (default: 500)

Reply with ONLY a Python comment block like:
# lr={{}}, gamma={{}}, epsilon_decay={{}}, episodes={{}}

PAPER CONTEXT:
{filtered_text}"""

    elif domain == "graph":
        prompt = f"""You are an expert GNN engineer. Extract ONLY the hyperparameters from the paper below.

I already have a working GCN model. I just need:
- learning_rate (default: 0.01)
- weight_decay (default: 5e-4)
- num_epochs (default: 200)
- dropout (default: 0.5)

Reply with ONLY a Python comment block like:
# lr={{}}, weight_decay={{}}, epochs={{}}, dropout={{}}

PAPER CONTEXT:
{filtered_text}"""

    elif domain == "algorithm":
        prompt = f"""You are an expert software engineer. Write a complete runnable Python script.

STRICT RULES:
- Implement the algorithm described in the context below
- Use only standard Python libraries (no ML frameworks needed)
- Add clear comments explaining each step
- Include at least 3 test cases with print statements showing results
- Print each test result: print("Test passed: ...")
- Always print: print(f"Algorithm Result: {{result}}")
- If anything is unclear write # UNKNOWN - assumed default

ALGORITHM CONTEXT:
{filtered_text}

Write the complete Python implementation now:"""

    else:
        # Fallback to ml
        domain = "ml"
        prompt = f"""You are an expert ML engineer. Extract ONLY the hyperparameters from the paper below.

Reply with ONLY a Python comment block like:
# lr={{}}, momentum={{}}, batch_size={{}}, epochs={{}}, weight_decay={{}}

PAPER CONTEXT:
{filtered_text}"""

    # ── Call Groq ─────────────────────────────────────────────────────────────
    response = llm.invoke(prompt)
    groq_output = response.content.strip()

    # Clean up markdown if present
    if "```python" in groq_output:
        groq_output = groq_output.split("```python")[1].split("```")[0]
    elif "```" in groq_output:
        groq_output = groq_output.split("```")[1].split("```")[0]

    # Warn if Groq didn't return a proper hyperparam comment block
    if domain != "algorithm":
        if not any(line.strip().startswith('#') for line in groq_output.splitlines()):
            print("⚠️  Groq did not return hyperparameters — using template defaults")

    # ── Inject known-working template + Groq hyperparams ─────────────────────
    if domain == "ml":
        code = ML_TEMPLATE
        # Try to patch hyperparams from Groq's comment if present
        code = _patch_hyperparams_ml(code, groq_output)

    elif domain == "nlp":
        code = NLP_TEMPLATE
        code = _patch_hyperparams_nlp(code, groq_output)

    elif domain == "recommendation":
        code = RECOMMENDATION_TEMPLATE
        code = _patch_hyperparams_rec(code, groq_output)

    elif domain == "rl":
        code = RL_TEMPLATE
        code = _patch_hyperparams_rl(code, groq_output)

    elif domain == "graph":
        code = GRAPH_TEMPLATE
        code = _patch_hyperparams_graph(code, groq_output)

    elif domain == "algorithm":
        # Algorithm is fully Groq-generated — no template
        code = ALGORITHM_TEMPLATE + "\n" + groq_output

    # ── Global safety fix: hardcoded flatten ──────────────────────────────────
    if domain in ["ml", "else"]:
        code = fix_hardcoded_flatten(code)
        if "LazyLinear" in code:
            print("⚠️  Warning: hardcoded flatten detected — auto-fixed with nn.LazyLinear")

    # ── Global safety fix: deprecated torchtext ───────────────────────────────
    if "import torchtext" in code or "from torchtext" in code:
        code = code.replace("import torchtext", "# torchtext deprecated - removed")
        code = code.replace("from torchtext", "# torchtext deprecated - removed")
        print("⚠️  Warning: torchtext removed — deprecated library")

    print("✅ Code generated successfully!")
    return code


# ── Hyperparameter patching helpers ──────────────────────────────────────────
# These extract values from Groq's comment and patch the template.
# If Groq returns garbage, defaults in the template are used — always safe.

def _extract_param(text, key, default):
    """Extract a named param from Groq comment, return default if not found.
    Handles int, float, and scientific notation (e.g. 1e-3, 5e-4)."""
    match = re.search(rf'{key}\s*=\s*([0-9e\.\-\+]+)', text, re.IGNORECASE)
    if match:
        try:
            val = match.group(1)
            if 'e' in val.lower() or '.' in val:
                return float(val)
            return int(val)
        except Exception:
            return default
    return default


def _patch_hyperparams_ml(code, groq_output):
    lr           = _extract_param(groq_output, 'lr',           0.01)
    momentum     = _extract_param(groq_output, 'momentum',     0.9)
    batch_size   = int(_extract_param(groq_output, 'batch_size', 128))
    epochs       = int(_extract_param(groq_output, 'epochs',     5))
    weight_decay = _extract_param(groq_output, 'weight_decay', 1e-4)

    # Safety caps — prevent CPU timeouts and OOM in Docker
    epochs     = min(epochs, 10)        # max 10 epochs — ~10 min on CPU
    batch_size = max(batch_size, 64)    # min 64 — smaller = slower per epoch
    batch_size = min(batch_size, 512)   # max 512 — larger = OOM in Docker

    code = code.replace("lr=0.01",          f"lr={lr}")
    code = code.replace("momentum=0.9",     f"momentum={momentum}")
    code = code.replace("batch_size=128",   f"batch_size={batch_size}")
    code = code.replace("num_epochs = 5",   f"num_epochs = {epochs}")
    code = code.replace("weight_decay=1e-4",f"weight_decay={weight_decay}")
    print(f"📌 Patched ML params: lr={lr}, momentum={momentum}, batch_size={batch_size}, epochs={epochs} (capped at 10)")
    return code


def _patch_hyperparams_nlp(code, groq_output):
    lr         = _extract_param(groq_output, 'lr',         1e-3)
    batch_size = int(_extract_param(groq_output, 'batch_size', 64))
    epochs     = int(_extract_param(groq_output, 'epochs',     10))
    dropout    = _extract_param(groq_output, 'dropout',    0.3)

    epochs     = min(epochs, 15)      # max 15 epochs for NLP
    batch_size = max(batch_size, 32)
    batch_size = min(batch_size, 256) # max 256 — prevent OOM

    code = code.replace("lr=1e-3",        f"lr={lr}")
    code = code.replace("batch_size=64",  f"batch_size={batch_size}")
    code = code.replace("num_epochs = 10",f"num_epochs = {epochs}")
    code = code.replace("Dropout(0.3)",   f"Dropout({dropout})")
    print(f"📌 Patched NLP params: lr={lr}, batch_size={batch_size}, epochs={epochs} (capped at 15)")
    return code


def _patch_hyperparams_rec(code, groq_output):
    lr            = _extract_param(groq_output, 'lr',            1e-3)
    embedding_dim = int(_extract_param(groq_output, 'embedding_dim', 50))
    epochs        = int(_extract_param(groq_output, 'epochs',        5))
    batch_size    = int(_extract_param(groq_output, 'batch_size',    256))

    epochs     = min(epochs, 10)      # max 10 epochs
    batch_size = max(batch_size, 128)
    batch_size = min(batch_size, 512) # max 512 — prevent OOM

    code = code.replace("lr=1e-3",           f"lr={lr}")
    code = code.replace("embedding_dim=50",  f"embedding_dim={embedding_dim}")
    code = code.replace("num_epochs = 5",    f"num_epochs = {epochs}")
    code = code.replace("batch_size=256",    f"batch_size={batch_size}")
    print(f"📌 Patched Rec params: lr={lr}, embedding_dim={embedding_dim}, epochs={epochs} (capped at 10)")
    return code


def _patch_hyperparams_rl(code, groq_output):
    lr            = _extract_param(groq_output, 'lr',            1e-3)
    gamma         = _extract_param(groq_output, 'gamma',         0.99)
    epsilon_decay = _extract_param(groq_output, 'epsilon_decay', 0.995)
    episodes      = int(_extract_param(groq_output, 'episodes',      500))

    episodes = min(episodes, 500)     # max 500 episodes — ~3 min on CPU

    code = code.replace("lr=1e-3",           f"lr={lr}")
    code = code.replace("gamma        = 0.99",  f"gamma        = {gamma}")
    code = code.replace("epsilon_decay= 0.995", f"epsilon_decay= {epsilon_decay}")
    code = code.replace("num_episodes = 500",   f"num_episodes = {episodes}")
    print(f"📌 Patched RL params: lr={lr}, gamma={gamma}, episodes={episodes} (capped at 500)")
    return code


def _patch_hyperparams_graph(code, groq_output):
    lr           = _extract_param(groq_output, 'lr',           0.01)
    weight_decay = _extract_param(groq_output, 'weight_decay', 5e-4)
    epochs       = int(_extract_param(groq_output, 'epochs',       200))
    dropout      = _extract_param(groq_output, 'dropout',      0.5)

    epochs = min(epochs, 200)         # max 200 epochs — fast since synthetic graph

    code = code.replace("lr=0.01",           f"lr={lr}")
    code = code.replace("weight_decay=5e-4", f"weight_decay={weight_decay}")
    code = code.replace("range(200)",        f"range({epochs})")
    code = code.replace("Dropout(0.5)",      f"Dropout({dropout})")
    print(f"📌 Patched Graph params: lr={lr}, weight_decay={weight_decay}, epochs={epochs} (capped at 200)")
    return code