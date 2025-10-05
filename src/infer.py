import torch
import tiktoken
from model import GPTConfig, GPTModel

CKPT_PATH = "ckpt.pt"

def load_model(ckpt_path=CKPT_PATH, device="cuda"):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    cfg = GPTConfig(
        vocab_size=vocab_size,
        n_layer=12,
        n_head=12,
        d_model=768,
        seq_len=1024
    )
    model = GPTModel(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, enc

def generate(model, enc, prompt, max_new_tokens=100, temperature=1.0, top_k=50, device="cuda"):
    model.eval()
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]  # (1, T)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(x)  # (1, T, vocab)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # top-k sampling
                v, ix = torch.topk(logits, top_k)
                mask = logits < v[:, [-1]]
                logits[mask] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    out = enc.decode(x[0].tolist())
    return out

if __name__ == "__main__":
    model, enc = load_model()
    prompt = "I am okay"
    out = generate(model, enc, prompt, max_new_tokens=100, temperature=0.8)
    print(out)
