import sys
import os
import json
import math
import re

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch")
    sys.exit(1)

MODEL_DIR = "alta-model"

# ── Tokenizer ─────────────────────────────────────────────────────────────────

TOKEN_RE = re.compile(r'<\|end\|>|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]|\s+')

def load_vocab(model_dir):
    path = os.path.join(model_dir, "vocab.json")
    with open(path) as f:
        data = json.load(f)
    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos

def encode(text, stoi):
    unk = stoi["<unk>"]
    return [stoi.get(tok, unk) for tok in TOKEN_RE.findall(text)]

def decode(ids, itos):
    return "".join(itos.get(i, "?") for i in ids)

# ── Model ─────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out     = nn.Linear(d_model, d_model, bias=False)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = math.sqrt(self.d_head)
        attn  = (q @ k.transpose(-2, -1)) / scale
        attn  = attn.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        attn  = torch.softmax(attn, dim=-1)
        attn  = self.drop(attn)
        out   = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, self.mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, max_new, stoi, itos, temperature=0.8):
        end_id = stoi.get("<|end|>", -1)
        for _ in range(max_new):
            idx_cond = idx[:, -self.max_len:]
            logits   = self(idx_cond)[:, -1, :]
            if temperature > 0:
                logits  = logits / temperature
                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)
            if next_id.item() == end_id:
                break
            idx = torch.cat([idx, next_id], dim=1)
        result = decode(idx[0].tolist(), itos)
        if "<|end|>" in result:
            result = result[:result.index("<|end|>")]
        return result

# ── Load ──────────────────────────────────────────────────────────────────────

def load_model(model_dir, device):
    config_path = os.path.join(model_dir, "config.json")
    model_path  = os.path.join(model_dir, "model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"ERROR: Model files not found in {model_dir}")
        print("Make sure the USB drive is plugged in and you've completed at least one training run.")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    model = MiniTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        max_len=cfg["max_len"],
        dropout=0.0,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ── Chat loop ─────────────────────────────────────────────────────────────────

def chat(model, stoi, itos, device, temperature=0.8, max_new=300):
    print("\nAlta is ready. Type your message and press Enter.")
    print("Commands: :temp <0-2> to change temperature, :quit to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.startswith(":quit"):
            print("Goodbye.")
            break

        if user_input.startswith(":temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"  Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("  Usage: :temp <float>  e.g. :temp 0.7")
            continue

        ctx    = f"### Prompt:\n{user_input}\n### Response:\n"
        ids    = torch.tensor([encode(ctx, stoi)], dtype=torch.long).to(device)
        output = model.generate(ids, max_new, stoi, itos, temperature=temperature)
        reply  = output[len(ctx):].strip()

        print(f"Alta: {reply}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Loading model from {MODEL_DIR}...")

    stoi, itos = load_vocab(MODEL_DIR)
    model      = load_model(MODEL_DIR, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {n_params:,} parameters")

    chat(model, stoi, itos, device)


if __name__ == "__main__":
    main()
