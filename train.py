import sys
import os
import json
import math
import random
import argparse
import glob
import re

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Run: pip install torch pandas tqdm")
    print("NOTE: PyTorch requires Python <=3.12. If you're on 3.14, run:")
    print("  pyenv install 3.12.9 && pyenv local 3.12.9 && pip install torch pandas tqdm")
    sys.exit(1)

try:
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing deps. Run: pip install torch pandas tqdm")
    sys.exit(1)
    
## get the operating system and set path to data accordingly
if os.name == "nt":  # Windows
    FULL_PATH = "data_windows.csv"
elif os.name == "posix":  # Unix/Linux/MacOS
    FULL_PATH = "/Volumes/USBDRIVE/"
else:
    print("ERROR: Unsupported operating system.")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

SEED        = 42
EPOCHS      = 2
BATCH_SIZE  = 8
MAX_LEN     = 512
STRIDE      = 256
N_LAYERS    = 4
N_HEADS     = 8
D_MODEL     = 256
D_FF        = 1024
DROPOUT     = 0.1
LR          = 3e-4
WEIGHT_DECAY= 0.01
WARMUP_STEPS= 100
GRAD_CLIP   = 1.0
SAVE_EVERY  = 1
GEN_TEMP    = 0.8
GEN_MAX_NEW = 200

DATA_PATH   = "data.csv"
MODEL_DIR   = FULL_PATH + "alta-model"
CKPT_DIR    = "checkpoints"

# ── Setup ─────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Tokenizer ─────────────────────────────────────────────────────────────────

# Splits text into: identifiers/words, numbers, punctuation, whitespace runs.
# Whitespace is kept as tokens so decode() reconstructs the original exactly.
TOKEN_RE = re.compile(r'<\|end\|>|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\w\s]|\s+')

def tokenize(text):
    return TOKEN_RE.findall(text)

def build_vocab(texts):
    tokens = set()
    for text in texts:
        tokens.update(tokenize(text))
    special = ["<pad>", "<unk>", "<|end|>"]
    vocab = special + sorted(tokens - set(special))
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    unk = stoi["<unk>"]
    return [stoi.get(tok, unk) for tok in tokenize(text)]

def decode(ids, itos):
    return "".join(itos.get(i, "?") for i in ids)

# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, df, stoi, max_len, stride):
        end_tok = "<|end|>"
        samples = []
        for _, row in df.iterrows():
            text = (
                f"### Prompt:\n{row['prompt']}\n"
                f"### Response:\n{row['completion']}{end_tok}"
            )
            samples.append(text)

        full = "\n".join(samples)
        ids = encode(full, stoi)

        self.chunks = []
        for start in range(0, len(ids) - max_len, stride):
            chunk = ids[start : start + max_len + 1]
            if len(chunk) == max_len + 1:
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]

# ── Model ─────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
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

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
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

        # causal mask buffer — filled lazily to device
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
        x    = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new, stoi, itos, temperature=0.8):
        end_id = stoi.get("<|end|>", -1)
        for _ in range(max_new):
            idx_cond = idx[:, -self.max_len:]
            logits   = self(idx_cond)[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                probs  = torch.softmax(logits, dim=-1)
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

# ── LR Scheduler ──────────────────────────────────────────────────────────────

def get_lr(step, d_model, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, 10000 - warmup)
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

# ── Main ──────────────────────────────────────────────────────────────────────

def find_latest_checkpoint():
    ckpts = glob.glob(os.path.join(CKPT_DIR, "epoch_*.pt"))
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]), reverse=True)
    for path in ckpts:
        try:
            torch.load(path, map_location="cpu")
            epoch = int(os.path.splitext(os.path.basename(path))[0].split("_")[1])
            return path, epoch
        except Exception:
            print(f"  [warning] corrupted checkpoint, skipping: {os.path.basename(path)}")
    return None, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=int, metavar="N", help="Resume from latest checkpoint and train N more epochs")
    args = parser.parse_args()

    print("\n── Loading data ──────────────────────────────────────")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} training pairs loaded")

    all_text = []
    for _, row in df.iterrows():
        all_text.append(f"### Prompt:\n{row['prompt']}\n### Response:\n{row['completion']}<|end|>")

    print("\n── Building vocabulary ───────────────────────────────")
    stoi, itos = build_vocab(all_text)
    vocab_size = len(stoi)
    print(f"  Vocabulary size: {vocab_size} characters")

    with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
        json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f)

    config = {
        "vocab_size": vocab_size,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_len": MAX_LEN,
        "dropout": DROPOUT,
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\n── Building dataset ──────────────────────────────────")
    dataset = TokenDataset(df, stoi, MAX_LEN, STRIDE)
    print(f"  {len(dataset)} training chunks")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "mps"),
    )

    print("\n── Initialising model ────────────────────────────────")
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(ignore_index=stoi["<pad>"])

    start_epoch = 1
    end_epoch   = EPOCHS
    global_step = 0

    if args.resume is not None:
        ckpt_path, resumed_epoch = find_latest_checkpoint()
        if ckpt_path is None:
            print("  No checkpoints found — starting fresh.")
        else:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                global_step = ckpt.get("global_step", 0)
            else:
                model.load_state_dict(ckpt)
                global_step = resumed_epoch * len(loader)
            start_epoch = resumed_epoch + 1
            end_epoch   = resumed_epoch + args.resume
            print(f"  Resumed from {ckpt_path} (epoch {resumed_epoch}, step {global_step})")
            print(f"  Training epochs {start_epoch} → {end_epoch}")

    print("\n── Training ──────────────────────────────────────────\n")

    epoch_bar = tqdm(range(start_epoch, end_epoch + 1), desc="Epochs", unit="ep", position=0)

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        batch_bar = tqdm(
            loader,
            desc=f"  Epoch {epoch:02d}",
            unit="batch",
            position=1,
            leave=False,
        )

        for x, y in batch_bar:
            x, y = x.to(device), y.to(device)

            # LR warmup / cosine decay
            lr_scale = get_lr(global_step, D_MODEL, WARMUP_STEPS)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale

            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), y.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss  += loss.item()
            global_step += 1

            batch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{LR * lr_scale:.2e}")

        avg_loss = total_loss / len(loader)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            }, ckpt_path)
            tqdm.write(f"  [checkpoint] saved → {ckpt_path}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
    print(f"\n── Model saved → {MODEL_DIR}/model.pt")

    # ── Inference test ────────────────────────────────────────────────────────

    print("\n── Sample generations ────────────────────────────────\n")
    model.eval()

    prompts = [
        "Write a Python function to reverse a string.",
        "Hey, how's it going?",
        "How do you reverse a list in JavaScript?",
    ]

    for p in prompts:
        ctx   = f"### Prompt:\n{p}\n### Response:\n"
        ids   = torch.tensor([encode(ctx, stoi)], dtype=torch.long).to(device)
        out   = model.generate(ids, GEN_MAX_NEW, stoi, itos, temperature=GEN_TEMP)
        reply = out[len(ctx):]
        print(f"Prompt:   {p}")
        print(f"Response: {reply.strip()}")
        print()


if __name__ == "__main__":
    main()
