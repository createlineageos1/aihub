import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re

chars = sorted(set("0123456789+-*/^= "))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

def encode(text):
    return [char2idx[c] for c in text]

def decode(indices):
    return "".join(idx2char[i] for i in indices)

class MathDataset(Dataset):
    def __init__(self, data):
        self.data = [encode(line) for line in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs)
    ys = nn.utils.rnn.pad_sequence(ys)
    return xs, ys

class MathTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x
      
data = [
    "2+3=5",
    "10-4=6",
    "3*4=12",
    "9/3=3",
    "2^3=8",
    "5+7=12",
    "8-3=5",
    "6*7=42",
    "12/4=3",
    "4^2=16"
]

dataset = MathDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

vocab_size = len(char2idx)
model = MathTransformer(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

epochs = 200
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")

def solve_math_expr(expr):
    try:
        expr = expr.replace("^", "**")
        if re.fullmatch(r"[0-9+\-*/^. ()]+", expr):
            return str(eval(expr))
    except:
        pass
    return None
  
def chat_predict(model, input_text, max_len=20):
    answer = solve_math_expr(input_text.strip("= "))
    if answer is not None:
        return answer

    model.eval()
    seq = encode(input_text)
    input_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(1)
    generated = seq[:]
    
    with torch.no_grad():
        for _ in range(max_len - len(seq)):
            output = model(input_seq)
            next_token_logits = output[-1, 0]
            next_token = torch.argmax(next_token_logits).item()
            generated.append(next_token)
            if idx2char[next_token] == " ":
                break
            input_seq = torch.tensor(generated, dtype=torch.long).unsqueeze(1)
    return decode(generated)

print("Chat with Victus 1M")
while True:
    user_input = input("Question: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = chat_predict(model, user_input)
    print("Response:", response)
