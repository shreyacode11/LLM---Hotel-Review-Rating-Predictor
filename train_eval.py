"""Classification head, training, evaluation, checkpointing and classify() (Ch 6 style)."""
import os, torch, torch.nn as nn, torch.nn.functional as F
import config as C, gpt_model as G
from data_prep import clean_text


def build_classifier(num_classes, device, unfreeze_last_n=C.UNFREEZE_LAST_N):
    """Pretrained GPT-2 124M -> freeze everything -> new Linear(768, num_classes) head
    -> unfreeze the last transformer block(s) and the final LayerNorm."""
    model = G.build_pretrained_gpt(dict(C.GPT_CONFIG_124M), "cpu")
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(C.SEED)                       # reproducible head initialisation
    model.out_head = nn.Linear(C.GPT_CONFIG_124M["emb_dim"], num_classes)
    for blk in model.trf_blocks[-unfreeze_last_n:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    return model.to(device)


def load_classifier(num_classes, path, device="cpu"):
    """Rebuild the architecture (no download needed) and load the fine-tuned checkpoint."""
    model = G.GPTModel(dict(C.GPT_CONFIG_124M))
    model.out_head = nn.Linear(C.GPT_CONFIG_124M["emb_dim"], num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def class_logits(model, x, lengths):
    """Logits of the LAST NON-PAD token of every review (causal attention => it has seen the
    whole review, and later padding cannot influence it)."""
    out = model(x)                                        # (batch, seq, num_classes)
    return out[torch.arange(x.size(0), device=x.device), lengths - 1]


def calc_loss_batch(batch, model, device):
    x, n, y = [t.to(device) for t in batch]
    return F.cross_entropy(class_logits(model, x, n), y)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x, n, y = [t.to(device) for t in batch]
        probs.append(torch.softmax(class_logits(model, x, n), dim=-1).cpu())
        labels.append(y.cpu())
    probs = torch.cat(probs)
    return probs.argmax(-1).numpy(), torch.cat(labels).numpy(), probs.numpy()


@torch.no_grad()
def calc_loss_loader(loader, model, device, num_batches=None):
    model.eval()
    n = len(loader) if num_batches is None else min(num_batches, len(loader))
    tot = 0.0
    for i, batch in enumerate(loader):
        if i >= n:
            break
        tot += calc_loss_batch(batch, model, device).item()
    return tot / n


def compute_metrics(y_true, y_pred, num_classes):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(y_true, y_pred)),     # in class-index units
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()}


def train_model(model, train_loader, val_loader, device, epochs, eval_freq, eval_iter, ckpt_path):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    hist = {"steps": [], "train": [], "val": [], "epoch_val_acc": []}
    step, best_val = -1, float("inf")
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = calc_loss_batch(batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % eval_freq == 0:
                tr = calc_loss_loader(train_loader, model, device, eval_iter)
                va = calc_loss_loader(val_loader, model, device, eval_iter)
                model.train()
                hist["steps"].append(step); hist["train"].append(tr); hist["val"].append(va)
                print(f"Ep {ep+1} step {step:05d}: train {tr:.3f} val {va:.3f}")
        yp, yt, _ = predict_loader(model, val_loader, device)
        acc = float((yp == yt).mean())
        hist["epoch_val_acc"].append(acc)
        print(f"== epoch {ep+1} validation accuracy {acc:.4f}")
        vl = calc_loss_loader(val_loader, model, device)
        if vl < best_val:                       # keep the checkpoint with the lowest val loss
            best_val = vl
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    return hist


def classify(text, model, tok, device="cpu", max_length=C.MAX_LENGTH):
    """classify(text) -> (class_index, probabilities). Same cleaning/truncation as training."""
    ids = tok.encode(clean_text(text))[:max_length]
    if not ids:
        return None, None
    x = torch.tensor([ids], device=device)
    n = torch.tensor([len(ids)], device=device)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(class_logits(model, x, n), dim=-1)[0]
    return int(p.argmax()), p.tolist()
