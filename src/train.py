# src/train.py
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import amp
import tiktoken

from model import GPTConfig, GPTModel
from data import TokenBinDataset  


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 100
CKPT_PATH = "ckpt.pt"


# ==== Training routine ====
def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scaler,
    epochs=5,
    device="cuda",
):
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # ----- training -----
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)  # (B, T, V)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (i + 1) % PRINT_EVERY == 0:
                avg = epoch_loss / (i + 1)
                print(f"Epoch {epoch} step {i+1}/{len(train_loader)} — train loss: {avg:.4f}")

        t1 = time.time()
        train_avg = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} — avg train loss: {train_avg:.4f} — time: {t1-t0:.1f}s")

        # ----- validation -----
        val_loss, val_ppl = evaluate(model, val_loader, loss_fn, device=device)
        print(f"Epoch {epoch} — val loss: {val_loss:.4f} | perplexity: {val_ppl:.2f}")

        # ----- checkpoint -----
        torch.save({
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, CKPT_PATH)
        print(f"Checkpoint saved to {CKPT_PATH}")


def evaluate(model, loader, loss_fn, device="cuda"):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            count += 1
    avg_loss = total_loss / count
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def main():
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    seq_len = 1024  

    # model config (124M params)
    cfg = GPTConfig(
        vocab_size=vocab_size,
        n_layer=12,
        n_head=12,
        d_model=768,
        seq_len=seq_len
    )
    model = GPTModel(cfg).to(DEVICE)

    print("Device:", DEVICE)
    print("Param count:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

   
    train_ds = TokenBinDataset("data/openwebtext_25B/train.bin", seq_len=1024)
    val_ds   = TokenBinDataset("data/openwebtext_25B/val.bin", seq_len=1024)


    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    # optimizer + amp scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    scaler = amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # train
    train_loop(model, train_loader, val_loader, optimizer, scaler, epochs=3, device=DEVICE)

    print("Training finished.")


if __name__ == "__main__":
    main()
