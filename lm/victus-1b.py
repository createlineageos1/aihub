import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT2(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_layer=2, n_head=4, block_size=64):
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

if __name__ == "__main__":
    vocab_size = 1000
    model = MiniGPT2(vocab_size)
    dummy_input = torch.randint(0, vocab_size, (1, 64))
    logits = model(dummy_input)
    print(logits.shape)
import torch
import torch.nn.functional as F

def sample(model, start_tokens, steps, temperature=1.0):
    model.eval()
    idx = torch.tensor(start_tokens).unsqueeze(0)
    for _ in range(steps):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx[0].tolist()
start = [42, 7, 13]
generated = sample(model, start, steps=20)
print("Generated:", generated)
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
def chat(model, tokenizer, steps=100):
    print("Hello! I'm Victus-1B, a tiny language model😾")
    while True:
        inp = input("Sen: ")
        if inp.lower() == "çık":
            break
        input_ids = tokenizer.encode(inp)
        output_ids = sample(model, input_ids, steps=steps)
        response = tokenizer.decode(output_ids[len(input_ids):])
        print("Bot:", response)
def sample(model, start_tokens, steps, temperature=1.0):
    model.eval()
    idx = torch.tensor(start_tokens).unsqueeze(0)
    for _ in range(steps):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return idx[0].tolist()
if __name__ == "__main__":
    text = "hello"
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    model = MiniGPT2(vocab_size)
    chat(model, tokenizer, steps=50)
