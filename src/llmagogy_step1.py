"""
llmagogy_step1.py

Baseline experiment: traditional pre-training on the full dataset (train_cum_5.txt).
This establishes a reference loss to compare against the multi-stage LLMagogy approach.
"""

import os
import sys
import time
import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Configuration
# ----------------------------

# Paths (relative to project root)
DATA_DIR = "../data/processed"
TRAIN_FILE = os.path.join(DATA_DIR, "train_cum_5.txt")
VAL_FILE = os.path.join(DATA_DIR, "val.txt")

# Hyperparameters
seed = 1337
torch.manual_seed(seed)
random.seed(seed)

# Force CUDA — no fallback to CPU
device = "cuda"

# Model
n_layer = 10
n_embd = 192
n_head = 6
block_size = 2048
dropout = 0.1
vocab_size = None  # will be set from data

# Training
batch_size = 4
max_iters = 300000
eval_interval = 2000
eval_iters = 800
learning_rate = 3e-4
grad_clip = 1.0


# ----------------------------
# Data loading
# ----------------------------

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
        return x, y

def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    return encode(text), vocab_size, stoi

# Load training data (defines the vocabulary)
train_tokens, vocab_size, stoi = load_data(TRAIN_FILE)

# Load validation data using the SAME vocabulary
with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_text = f.read()
# Encode using train's stoi; skip unknown characters
val_tokens = [stoi[c] for c in val_text if c in stoi]

# Optional: warn about missing characters
val_chars = set(val_text)
train_chars = set(stoi.keys())
missing = val_chars - train_chars
if missing:
    print(f"⚠️ Warning: {len(missing)} characters in val not in train vocab. Examples: {sorted(list(missing))[:5]}")

# Build datasets
train_dataset = TextDataset(train_tokens, block_size)
val_dataset = TextDataset(val_tokens, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# ----------------------------
# Model (minimal GPT)
# ----------------------------

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.n_head, self.head_dim).transpose(1, 2), qkv)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# ----------------------------
# Training loop
# ----------------------------

start_time = time.time()
duration = 0
torch.cuda.empty_cache()
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def estimate_loss():
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        num_batches = min(eval_iters, len(loader))
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / num_batches
    model.train()
    return losses

# Training
print(f"Model parameters: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}, block_size={block_size}, dropout={dropout}, batch_size={batch_size}, max_iters={max_iters}")
model_size = (vocab_size * n_embd + n_embd * block_size + n_layer * (12 * n_embd**2 + 13 * n_embd) + n_embd * vocab_size) / 1000000
print(f"Model size: vocab_size={vocab_size}, model_size={model_size:.2f}M parameters")
print(f"Starting baseline training on {device}...")
model.train()
train_iter = iter(train_loader)
losses = None
for iter_num in range(max_iters):
    # Get batch
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    x, y = x.to(device), y.to(device)

    # Forward
    logits, loss = model(x, y)

    # Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Logging
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        duration += time.time() - start_time
        prev_losses = losses
        losses = estimate_loss()
        if prev_losses is None:
            print(f"step {iter_num}: total duration {duration:.0f} seconds, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val/train {losses['val']/losses['train']:.4f}")
        else:
            print(f"step {iter_num}: total duration {duration:.0f} seconds, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val/train {losses['val']/losses['train']:.4f}, \
train diff {(losses['train']-prev_losses['train'])/prev_losses['train']*100:.2f}%, val diff {(losses['val']-prev_losses['val'])/prev_losses['val']*100:.2f}%")
        start_time = time.time()
     
