import os
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F

input_file = '../data/input.txt'

training = True

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
save_interval = 500
save_dir = '../checkpoints'
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
lr = 3e-4

using_gpu = True

device = torch.device("cpu")
if using_gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

torch.manual_seed(1337)

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# number of unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

# build mapping between chars and ints
dict_stoi = {ch: i for i, ch in enumerate(chars)}
dict_itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [dict_stoi[c] for c in s]
decode = lambda l: ''.join([dict_itos[i] for i in l])

# train test splitting
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad
def estimate_loss():
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

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        score = q @ k.transpose(-2, -1) * C ** -0.5 # score: (B, T, T)
        score = score.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)

        out = score @ v
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B,T,C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is B*T array of current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last timestep
            logits = logits[:, -1, :]
            # apply softmax to get largest probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel().to(device)

idx = torch.zeros((1, 1), dtype=torch.long)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32

if not training:
    model.load_state_dict(os.path.join(save_dir, 'v2.pth'))
else:
    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
        
        # sample a batch of data
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), os.path.join(save_dir, 'v2.pth'))

# print(loss.item())
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
