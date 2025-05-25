import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class MiniGPT2(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_layer=4, n_head=8, block_size=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4*n_embd,
            activation='gelu'
        ) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.n_embd = n_embd

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = tok_emb + pos_emb
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids):
        return ''.join([self.itos.get(i, '?') for i in ids])

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

def train(model, optimizer, data, epochs=3000, batch_size=32, block_size=128, eval_interval=200):
    model.train()
    for step in range(epochs):
        xb, yb = get_batch(data, batch_size, block_size)
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step} - Loss: {loss.item():.4f}")

def sample(model, start_tokens, steps, temperature=0.7):
    model.eval()
    idx = torch.tensor(start_tokens).unsqueeze(0)
    with torch.no_grad():
        for _ in range(steps):
            idx_cond = idx[:, -model.block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
    return idx[0].tolist()

def chat(model, tokenizer, steps=100):
    print("Hi! I'm Victus-1B!")
    while True:
        inp = input("You: ")
        if inp.lower() == "çık":
            break
        input_ids = tokenizer.encode(inp)
        output_ids = sample(model, input_ids, steps=steps)
        response = tokenizer.decode(output_ids[len(input_ids):])
        print("Bot:", response)

if __name__ == "__main__":
    text = open("veri.txt", encoding="utf8").read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model = MiniGPT2(vocab_size=tokenizer.vocab_size, n_embd=256, n_layer=4, n_head=8, block_size=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("Training the model...")
    train(model, optimizer, data, epochs=3000, batch_size=32, block_size=128, eval_interval=200)
    print("Model is trained, starting the chat...")

    chat(model, tokenizer, steps=100)
