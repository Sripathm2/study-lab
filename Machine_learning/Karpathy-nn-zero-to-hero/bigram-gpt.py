"""
Decoder-only transformer (GPT) — character-level language model.

Karpathy, "Let's build GPT" (Zero to Hero #7). Trains on a plain-text corpus
and generates Shakespeare-like text one character at a time.

Architecture: token + positional embeddings -> n_layer transformer blocks
(masked multi-head self-attention + feed-forward, each wrapped in a residual
connection and preceded by a LayerNorm) -> final LayerNorm -> linear head to
vocabulary logits. Trained with cross-entropy on next-token prediction.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size = 16        # independent sequences processed in parallel
block_size = 32        # maximum context length for predictions
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cpu'         # 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64            # embedding dimension
n_head = 4             # attention heads per block
n_layer = 4            # number of transformer blocks
dropout = 0.0
# ---------------------------------------------------------------------------

torch.manual_seed(1337)

# Get the corpus with e.g.:
#   wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('../data/gpt-input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenizer: a lossless, reversible text <-> ints mapping.
# (Production models use subword tokenizers instead: SentencePiece, tiktoken/BPE.)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]           # string -> list[int]
decode = lambda l: ''.join(itos[i] for i in l)    # list[int] -> string

# Train / val split: first 90% trains, last 10% validates.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Sample a random batch of (x, y); y is x shifted one position left."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Average train/val loss over eval_iters batches (less noisy than one batch)."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of masked self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is a constant (not a learned parameter), so register it as a buffer.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) -> out: (B, T, head_size)
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # Attention scores ("affinities"), scaled by 1/sqrt(head_size) so the
        # softmax stays diffuse instead of saturating toward one-hot.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5          # (B, T, T)
        # Causal mask: a token may only attend to itself and the past.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                  # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of the values.
        v = self.value(x)  # (B, T, hs)
        out = wei @ v      # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Several self-attention heads running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection back into the residual stream: linearly recombine the
        # concatenated per-head outputs before they are added to x.
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate heads
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Per-token MLP: the 'computation' step after attention's 'communication'."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # widen (the 4x factor follows the paper)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # project back down to the residual dim
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (feed-forward)."""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of attention heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # pre-norm: LayerNorm before each sub-block
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # The `x + ...` are residual connections (a direct shortcut back to the input).
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """Decoder-only transformer. (Name kept from the bigram build-up; this is now a full GPT.)"""

    def __init__(self):
        super().__init__()
        # Token embedding: each token's vector. Positional embedding: each slot's vector.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)            # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx, targets: (B, T) integer tensors
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                 # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))   # (T, C)
        x = tok_emb + pos_emb   # token identity + position
        x = self.blocks(x)      # tokens communicate, then each computes
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T) indices of the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]         # crop to the last block_size tokens
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]               # take the last time step -> (B, C)
            probs = F.softmax(logits, dim=-1)       # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# AdamW: adaptive per-parameter step sizes with decoupled weight decay.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every so often, report averaged train/val loss.
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the trained model, starting from a single (zero) token.
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))