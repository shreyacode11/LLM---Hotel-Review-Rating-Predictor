"""Course-style GPTModel (Ch 3-4) and load_weights_into_gpt (Ch 5).
No Hugging Face model classes are used."""
import numpy as np, torch, torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out, self.num_heads, self.head_dim = d_out, num_heads, d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, n, _ = x.shape
        q = self.W_query(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(2, 3)
        scores.masked_fill_(self.mask.bool()[:n, :n], -torch.inf)
        w = torch.softmax(scores / self.head_dim ** 0.5, dim=-1)
        w = self.dropout(w)
        ctx = (w @ v).transpose(1, 2).contiguous().view(b, n, self.d_out)
        return self.out_proj(ctx)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
                                      cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1, self.norm2 = LayerNorm(cfg["emb_dim"]), LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        b, n = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(n, device=idx.device))
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))


def _assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return nn.Parameter(torch.as_tensor(np.asarray(right)).clone())


def load_weights_into_gpt(gpt, p):
    """Copy a GPT-2 parameter dict `p` (OpenAI/TF-style layout, numpy arrays) into `gpt`."""
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, p["wpe"])
    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, p["wte"])
    for b in range(len(p["blocks"])):
        blk, t = gpt.trf_blocks[b], p["blocks"][b]
        qw, kw, vw = np.split(t["attn"]["c_attn"]["w"], 3, axis=-1)
        blk.att.W_query.weight = _assign(blk.att.W_query.weight, qw.T)
        blk.att.W_key.weight = _assign(blk.att.W_key.weight, kw.T)
        blk.att.W_value.weight = _assign(blk.att.W_value.weight, vw.T)
        qb, kb, vb = np.split(t["attn"]["c_attn"]["b"], 3, axis=-1)
        blk.att.W_query.bias = _assign(blk.att.W_query.bias, qb)
        blk.att.W_key.bias = _assign(blk.att.W_key.bias, kb)
        blk.att.W_value.bias = _assign(blk.att.W_value.bias, vb)
        blk.att.out_proj.weight = _assign(blk.att.out_proj.weight, t["attn"]["c_proj"]["w"].T)
        blk.att.out_proj.bias = _assign(blk.att.out_proj.bias, t["attn"]["c_proj"]["b"])
        blk.ff.layers[0].weight = _assign(blk.ff.layers[0].weight, t["mlp"]["c_fc"]["w"].T)
        blk.ff.layers[0].bias = _assign(blk.ff.layers[0].bias, t["mlp"]["c_fc"]["b"])
        blk.ff.layers[2].weight = _assign(blk.ff.layers[2].weight, t["mlp"]["c_proj"]["w"].T)
        blk.ff.layers[2].bias = _assign(blk.ff.layers[2].bias, t["mlp"]["c_proj"]["b"])
        blk.norm1.scale = _assign(blk.norm1.scale, t["ln_1"]["g"])
        blk.norm1.shift = _assign(blk.norm1.shift, t["ln_1"]["b"])
        blk.norm2.scale = _assign(blk.norm2.scale, t["ln_2"]["g"])
        blk.norm2.shift = _assign(blk.norm2.shift, t["ln_2"]["b"])
    gpt.final_norm.scale = _assign(gpt.final_norm.scale, p["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, p["b"])
    # Output head initialised from wte as an untied copy (as in the course);
    # it is replaced by a classification head in train_eval.build_classifier().
    gpt.out_head.weight = _assign(gpt.out_head.weight, p["wte"])


def load_gpt2_params_from_safetensors(n_layers=12):
    """Download the public GPT-2 124M weights file (model.safetensors) with huggingface_hub
    and rearrange the tensors into the OpenAI-style dict expected by load_weights_into_gpt.
    Only the *file* is downloaded; no Hugging Face model class is used."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    sd = load_file(hf_hub_download("openai-community/gpt2", "model.safetensors"))
    g = lambda k: sd[k] if k in sd else sd["transformer." + k]
    p = {"wpe": g("wpe.weight"), "wte": g("wte.weight"),
         "g": g("ln_f.weight"), "b": g("ln_f.bias"), "blocks": []}
    for i in range(n_layers):
        h = f"h.{i}."
        p["blocks"].append({
            "attn": {"c_attn": {"w": g(h + "attn.c_attn.weight"), "b": g(h + "attn.c_attn.bias")},
                     "c_proj": {"w": g(h + "attn.c_proj.weight"), "b": g(h + "attn.c_proj.bias")}},
            "mlp": {"c_fc": {"w": g(h + "mlp.c_fc.weight"), "b": g(h + "mlp.c_fc.bias")},
                    "c_proj": {"w": g(h + "mlp.c_proj.weight"), "b": g(h + "mlp.c_proj.bias")}},
            "ln_1": {"g": g(h + "ln_1.weight"), "b": g(h + "ln_1.bias")},
            "ln_2": {"g": g(h + "ln_2.weight"), "b": g(h + "ln_2.bias")}})
    return p


def build_pretrained_gpt(cfg, device="cpu"):
    model = GPTModel(cfg)
    load_weights_into_gpt(model, load_gpt2_params_from_safetensors(cfg["n_layers"]))
    return model.to(device).eval()
