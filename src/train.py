# src/train.py
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import GPTConfig, GPTModel
import tiktoken

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 10
CKPT_PATH = "ckpt.pt"

# ==== Tiny Dataset (overfit target) ====
SAMPLE_TEXTS = [
    "Hello world! This is a tiny corpus to test overfitting.",
    "The quick brown fox jumps over the lazy dog.",
    "GPT models are fun to implement from scratch.",
    "I like building models and debugging them step by step.",
]

class TinyTextDataset(Dataset):
    def __init__(self, tokenizer, seq_len=128, repeats=256):
        # tokenize and make many short sequences by concatenation
        self.seq_len = seq_len
        enc = tokenizer
        raw = "\n".join(SAMPLE_TEXTS) + "\n"
        tokens = enc.encode(raw)
        tokens = tokens * repeats
        # chop into sequences (non-overlapping)
        self.examples = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i : i + seq_len]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    # batch: list of (seq_len,) tensors
    return torch.stack(batch, dim=0)

# ==== Training routine ====
def train_loop(
    model,
    dataloader,
    optimizer,
    scaler,
    epochs=5,
    device="cuda",
):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)  # (B, T)
            inputs = batch[:, :-1]   # predict next token
            targets = batch[:, 1:]
            B, T = inputs.shape

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits = model(inputs)  # (B, T, V)
                logits = logits.view(-1, logits.size(-1))  # (B*T, V)
                loss = loss_fn(logits, targets.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (i + 1) % PRINT_EVERY == 0:
                avg = epoch_loss / (i + 1)
                print(f"Epoch {epoch} step {i+1}/{len(dataloader)} — loss: {avg:.4f}")

        t1 = time.time()
        print(f"Epoch {epoch} ended — avg loss: {epoch_loss/len(dataloader):.4f} — time: {t1-t0:.1f}s")
        # save checkpoint each epoch
        torch.save({"model_state": model.state_dict(), "optimizer": optimizer.state_dict()}, CKPT_PATH)

def main():
    # tokenizer: use tiktoken directly for speed and GPT-2 vocab compatibility
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    seq_len = 128  # short for fast tests

    # model config (124M)
    cfg = GPTConfig(vocab_size=vocab_size, n_layer=12, n_head=12, d_model=768, seq_len=seq_len)
    model = GPTModel(cfg).to(DEVICE)

    print("Device:", DEVICE)
    print("Param count:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # tiny dataset and loader
    ds = TinyTextDataset(tokenizer=enc, seq_len=seq_len, repeats=64)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # optimizer + amp scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    train_loop(model, loader, optimizer, scaler, epochs=3, device=DEVICE)

    print("Training finished. Checkpoint saved to", CKPT_PATH)

if __name__ == "__main__":
    main()
