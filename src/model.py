import math
import torch
import torch.nn as nn
from typing import Optional

class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 12,
        n_head: int = 12,
        d_model: int = 768,
        seq_len: int = 1024,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.seq_len = seq_len
        self.ff_mult = ff_mult
        self.dropout = dropout

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_model = config.d_model
        n_head = config.n_head
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # x: (B, T, C)
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # reshape to (B, nh, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

        if attn_mask is not None:
            # attn_mask expected broadcastable to (B, nh, T, T)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (B, nh, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(attn_output)

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_model = config.d_model
        hidden = config.ff_mult * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        d_model = config.d_model
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff = FeedForward(config)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5)

        # LM head will tie weights with token embeddings (weight tying)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._tie_weights()

        # initialize weights
        self.apply(self._init_weights)

    def _tie_weights(self):
        # tie lm_head weight and token embedding weight
        self.lm_head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # normal init like GPT-2
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_causal_mask(self, T, device):
        # returns (1, 1, T, T) mask with ones in allow positions
        # using ones for allowed positions (so mask==0 are blocked)
        mask = torch.tril(torch.ones((T, T), dtype=torch.uint8, device=device))
        return mask.view(1, 1, T, T)  # broadcastable

    def forward(self, input_ids: torch.LongTensor):
        """
        input_ids: (B, T)
        returns logits: (B, T, vocab_size)
        """
        B, T = input_ids.size()
        assert T <= self.config.seq_len, f"Sequence length {T} > config.seq_len {self.config.seq_len}"

        token_embeddings = self.tok_emb(input_ids)          # (B, T, d_model)
        pos_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_emb(pos_ids)              # (1, T, d_model)
        x = token_embeddings + pos_embeddings
        x = self.drop(x)

        # causal mask
        attn_mask = self._build_causal_mask(T, device=input_ids.device)  # (1,1,T,T)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # tie to token embedding
        return logits

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    vocab_size = 50257  # GPT-2 vocab
    cfg = GPTConfig(vocab_size=vocab_size, n_layer=12, n_head=12, d_model=768, seq_len=1024)
    model = GPTModel(cfg)

    print("Model parameter count:", count_parameters(model))
    B, T = 2, 16
    input_ids = torch.randint(0, vocab_size, (B, T))
    logits = model(input_ids)
    print("Logits shape:", logits.shape)  # should be (B, T, vocab_size)
