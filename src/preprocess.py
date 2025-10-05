import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def preprocess_openwebtext(
    out_dir="data/openwebtext",
    train_ratio=0.9,
    target_tokens=20_000_000_000  
):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading OpenWebText...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    enc = tiktoken.get_encoding("gpt2")
    token_stream = []

    print("Tokenizing...")
    for i, example in enumerate(tqdm(dataset, desc="examples")):
        text = example["text"]
        tokens = enc.encode(text)
        tokens.append(enc.eot_token) 
        token_stream.extend(tokens)

        # stop early if we hit the token budget
        if target_tokens and len(token_stream) >= target_tokens:
            print(f"Reached target {target_tokens:,} tokens, stopping.")
            break

    n_total = len(token_stream)
    split = int(train_ratio * n_total)

    train_tokens = np.array(token_stream[:split], dtype=np.uint32)
    val_tokens   = np.array(token_stream[split:], dtype=np.uint32)

    train_tokens.tofile(os.path.join(out_dir, "train.bin"))
    val_tokens.tofile(os.path.join(out_dir, "val.bin"))

    print(f"Total tokens: {n_total:,}")
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")
    print(f"Saved to {out_dir}/train.bin and val.bin")

if __name__ == "__main__":
    # Trying with 25B tokens as I dont have compute to do full OWT
    preprocess_openwebtext(out_dir="data/openwebtext_25B", target_tokens=25_000_000_000)
