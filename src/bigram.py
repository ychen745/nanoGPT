import os
import shutil
import torch
import time
import torch.nn as nn
from torch.nn import functional as F

input_file = '../data/input.txt'

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
lr = 1e-2

using_gpu = False

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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

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
            # get predictions
            logits, loss = self(idx)
            # focus only on the last timestep
            logits = logits[:, -1, :]
            # apply softmax to get largest probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size).to(device)

idx = torch.zeros((1, 1), dtype=torch.long)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32

start_time = time.time()
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

# print(loss.item())
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
print(f"training time: {time.time() - start_time} seconds")
