"""
llmagogy_step5.py

Multi-stage pre-training: gradual model expansion on cumulative datasets.
This implements the core idea of LLMagogyV2: "growing" the model stage by stage.
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

# Paths
DATA_DIR = "../data/processed"
VAL_FILE = os.path.join(DATA_DIR, "val.txt")

# Fixed seed
seed = 1337
torch.manual_seed(seed)
random.seed(seed)

# Force CUDA
device = "cuda"

# Training
eval_interval1 = 2000
eval12_threshold = 100000
eval_interval2 = 2000
eval_iters = 800
learning_rate = 3e-4
grad_clip = 1.0
zero_new_weights = False

# Scheduler: 5 stages of model growth
# Format: (n_layer, n_embd, n_head, block_size, dropout, batch_size, max_iters, train file)
#scheduler = [
#    (2, 32, 1, 128, 0.0, 64, 40000, "train_cum_5.txt"),      # Stage 1
#    (4, 64, 2, 256, 0.05, 64, 45000, "train_cum_5.txt"),     # Stage 2
#    (6, 64, 2, 512, 0.05, 64, 30000, "train_cum_5.txt"),    # Stage 3
#    (9, 96, 3, 1024, 0.05, 32, 45000, "train_cum_5.txt"),    # Stage 4
#    (12, 128, 4, 2048, 0.1, 4, 300000, "train_cum_5.txt"),    # Stage 5 (final)
#]

#scheduler = [
#    (2, 64, 2, 128, 0.0, 64, 50000, "train_cum_5.txt"),      # Stage 1
#    (4, 96, 3, 256, 0.05, 64, 55000, "train_cum_5.txt"),     # Stage 2
#    (6, 128, 4, 512, 0.05, 64, 45000, "train_cum_5.txt"),    # Stage 3
#    (8, 160, 5, 1536, 0.1, 8, 50000, "train_cum_5.txt"),    # Stage 4
#    (10, 192, 6, 2048, 0.1, 4, 300000, "train_cum_5.txt"),    # Stage 5 (final)
#]

scheduler = [
    (2, 64, 2, 128, 0.0, 64, 50000, "train_cum_5.txt"),      # Stage 1
    (3, 96, 3, 256, 0.05, 64, 45000, "train_cum_5.txt"),     # Stage 2
    (4, 128, 4, 512, 0.05, 64, 35000, "train_cum_5.txt"),    # Stage 3
    (6, 192, 6, 1536, 0.1, 8, 65000, "train_cum_5.txt"),    # Stage 4
    (8, 256, 8, 2048, 0.15, 4, 300000, "train_cum_5.txt"),    # Stage 5 (final)
]


#scheduler = [
#    (2, 64, 2, 128, 0.0, 64, 280000, "train_cum_5.txt"),      # Stage 1
#    (4, 96, 3, 256, 0.05, 64, 100000, "train_cum_5.txt"),     # Stage 2
#    (6, 128, 4, 512, 0.05, 64, 75000, "train_cum_5.txt"),    # Stage 3
#    (8, 160, 5, 1536, 0.1, 8, 60000, "train_cum_5.txt"),    # Stage 4
#    (10, 192, 6, 2048, 0.1, 4, 300000, "train_cum_5.txt"),    # Stage 5 (final)
#]


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

def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]

# ----------------------------
# Model (same as step1)
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
# Model expansion utilities
# ----------------------------

def expand_model(old_model, new_config, vocab_size, zero_weights):
    n_layer_old = len(old_model.blocks)
    n_embd_old = old_model.tok_emb.embedding_dim
    block_size_old = old_model.block_size

    n_layer_new, n_embd_new, n_head_new, block_size_new, dropout_new, _, _, _ = new_config

    new_model = GPT(vocab_size, n_embd_new, n_head_new, n_layer_new, block_size_new, dropout_new)
    new_model.to(old_model.tok_emb.weight.device)

    # Expand token embedding
    min_embd = min(n_embd_old, n_embd_new)
    new_model.tok_emb.weight.data[:vocab_size, :min_embd] = old_model.tok_emb.weight.data[:vocab_size, :min_embd]
    if n_embd_new > n_embd_old:
        if zero_weights:
            nn.init.zeros_(new_model.tok_emb.weight.data[:, n_embd_old:])
        else:
            nn.init.normal_(new_model.tok_emb.weight.data[:, n_embd_old:], std=0.02)

    # Expand position embedding
    min_block = min(block_size_old, block_size_new)
    new_model.pos_emb.data[:, :min_block, :min_embd] = old_model.pos_emb.data[:, :min_block, :min_embd]
    if block_size_new > block_size_old or n_embd_new > n_embd_old:
        if zero_weights:
            nn.init.zeros_(new_model.pos_emb.data[:, min_block:, :])
            nn.init.zeros_(new_model.pos_emb.data[:, :, min_embd:])
        else:
            nn.init.normal_(new_model.pos_emb.data[:, min_block:, :], std=0.02)
            nn.init.normal_(new_model.pos_emb.data[:, :, min_embd:], std=0.02)

    # Expand blocks
    for i in range(min(n_layer_old, n_layer_new)):
        old_block = old_model.blocks[i]
        new_block = new_model.blocks[i]

        min_embd = min(n_embd_old, n_embd_new)
        new_block.ln1.weight.data[:min_embd] = old_block.ln1.weight.data[:min_embd]
        new_block.ln1.bias.data[:min_embd] = old_block.ln1.bias.data[:min_embd]
        new_block.ln2.weight.data[:min_embd] = old_block.ln2.weight.data[:min_embd]
        new_block.ln2.bias.data[:min_embd] = old_block.ln2.bias.data[:min_embd]

        expand_linear(old_block.attn.qkv, new_block.attn.qkv, n_embd_old, n_embd_new, zero_weights)
        expand_linear(old_block.attn.proj, new_block.attn.proj, n_embd_old, n_embd_new, zero_weights)
        expand_linear(old_block.mlp.c_fc, new_block.mlp.c_fc, n_embd_old, 4 * n_embd_new, zero_weights)
        expand_linear(old_block.mlp.c_proj, new_block.mlp.c_proj, 4 * n_embd_old, n_embd_new, zero_weights)

    # Final LayerNorm and head
    new_model.ln_f.weight.data[:min_embd] = old_model.ln_f.weight.data[:min_embd]
    new_model.ln_f.bias.data[:min_embd] = old_model.ln_f.bias.data[:min_embd]
    expand_linear(old_model.head, new_model.head, n_embd_old, vocab_size, zero_weights)

    return new_model

def expand_linear(old_layer, new_layer, old_in, new_out, zero_weights):
    old_weight = old_layer.weight.data
    new_weight = new_layer.weight.data

    in_features_old = old_weight.shape[1]
    out_features_old = old_weight.shape[0]
    in_features_new = new_weight.shape[1]
    out_features_new = new_weight.shape[0]

    actual_in = min(in_features_old, old_in)
    actual_out = min(out_features_old, new_out)

    new_layer.weight.data[:actual_out, :actual_in] = old_weight[:actual_out, :actual_in]

    if out_features_new > actual_out:
        if zero_weights:
            nn.init.zeros_(new_layer.weight.data[actual_out:, :])
        else:
            nn.init.normal_(new_layer.weight.data[actual_out:, :], std=0.02)
    if in_features_new > actual_in:
        if zero_weights:
            nn.init.zeros_(new_layer.weight.data[:, actual_in:])
        else:
            nn.init.normal_(new_layer.weight.data[:, actual_in:], std=0.02)

    if new_layer.bias is not None:
        if out_features_new > actual_out:
            nn.init.zeros_(new_layer.bias.data[actual_out:])
        new_layer.bias.data[:actual_out] = old_layer.bias.data[:actual_out]

# ----------------------------
# Training loop
# ----------------------------

def estimate_loss(model, train_loader, val_loader):
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

def main():
    print("Starting multi-stage LLMagogy training...")

    # Build unified vocabulary from FULL dataset
    full_train_file = os.path.join(DATA_DIR, "train_cum_5.txt")
    with open(full_train_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}

    model = None

    for stage_idx, config in enumerate(scheduler):
        duration = 0
        n_layer, n_embd, n_head, block_size, dropout, batch_size, max_iters, train_file = config
        model_size = (vocab_size * n_embd + n_embd * block_size + n_layer * (12 * n_embd**2 + 13 * n_embd) + n_embd * vocab_size) / 1000000
        train_file = os.path.join(DATA_DIR, train_file)
        print(f"\n=== Stage {stage_idx + 1} ===")
        print(f"Config: n_layer={n_layer}, n_embd={n_embd}, n_head={n_head}, block_size={block_size}, dropout={dropout}, batch_size={batch_size}, max_iters={max_iters}, zero_new_weights={zero_new_weights}")
        print(f"Model size: vocab_size={vocab_size}, model_size={model_size:.2f}M parameters")
        
        # Load current stage data using unified vocabulary
        with open(train_file, "r", encoding="utf-8") as f:
            train_text = f.read()
        train_tokens = encode(train_text, stoi)

        with open(VAL_FILE, "r", encoding="utf-8") as f:
            val_text = f.read()
        val_tokens = encode(val_text, stoi)

        train_dataset = TextDataset(train_tokens, block_size)
        val_dataset = TextDataset(val_tokens, block_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # Create or expand model
        torch.cuda.empty_cache()
        if stage_idx == 0:
            model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
        else:
            model = expand_model(model, config, vocab_size, zero_new_weights)

        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Training
        model.train()
        train_iter = iter(train_loader)
        start_time = time.time()
        losses = None
        for iter_num in range(max_iters):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if (stage_idx == 4 and ((iter_num % eval_interval1 == 0 and iter_num < eval12_threshold) or (iter_num % eval_interval2 == 0 and iter_num >= eval12_threshold))) or iter_num == max_iters - 1:
                duration += time.time() - start_time
                prev_losses = losses                
                losses = estimate_loss(model, train_loader, val_loader)
                if prev_losses is None:
                    print(f"step {iter_num}: total duration {duration:.0f} seconds, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val/train {losses['val']/losses['train']:.4f}")
                else:
                    print(f"step {iter_num}: total duration {duration:.0f} seconds, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val/train {losses['val']/losses['train']:.4f}, \
train diff {(losses['train']-prev_losses['train'])/prev_losses['train']*100:.2f}%, val diff {(losses['val']-prev_losses['val'])/prev_losses['val']*100:.2f}%")
                start_time = time.time()
        if stage_idx==4:
            break
    print("\n✅ Multi-stage training completed.")

if __name__ == "__main__":
    main()
