import os
import shutil
import torch
import torch.nn as nn
from torch.nn import functional as F

input_file = '/content/drive/MyDrive/nanoGPT/data/input.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# build dictionaries between chars and ints
dict_stoi = {ch: i for i, ch in enumerate(chars)}
dict_itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [dict_stoi[c] for c in s]
decode = lambda l: ''.join([dict_itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



batch_size = 32
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    return x, y

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

model = BigramLanguageModel(vocab_size)

idx = torch.zeros((1, 1), dtype=torch.long)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32

for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print(loss.item())
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
