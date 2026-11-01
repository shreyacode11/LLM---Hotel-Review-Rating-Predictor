"""Loading, cleaning, stratified subset/split, tokenisation, Dataset with truncation + padding."""
import re
import pandas as pd, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import config as C


def clean_text(t):
    t = re.sub(r"<[^>]+>", " ", str(t))      # strip any HTML tags
    return re.sub(r"\s+", " ", t).strip()    # collapse whitespace


def load_reviews(path=C.CSV_PATH):
    df = pd.read_csv(path).rename(columns={"Review": "text", "Rating": "rating"})[["text", "rating"]]
    stats = {"raw_rows": len(df)}
    df = df.dropna()
    df["text"] = df["text"].map(clean_text)
    df = df[df["rating"].between(1, 5) & (df["text"].str.split().str.len() >= 3)]
    stats["invalid_removed"] = stats["raw_rows"] - len(df)
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    stats["duplicates_removed"] = before - len(df)
    stats["clean_rows"] = len(df)
    stats["rating_counts"] = {int(k): int(v) for k, v in df["rating"].value_counts().sort_index().items()}
    return df, stats


def make_splits(df, n=C.SUBSET_SIZE, seed=C.SEED):
    """Stratified subset of n rows, then stratified 70/15/15 split (seeded)."""
    if n < len(df):
        df, _ = train_test_split(df, train_size=n, stratify=df["rating"], random_state=seed)
    train, rest = train_test_split(df, train_size=C.TRAIN_FRAC, stratify=df["rating"], random_state=seed)
    val, test = train_test_split(rest, train_size=C.VAL_FRAC / (1 - C.TRAIN_FRAC),
                                 stratify=rest["rating"], random_state=seed)
    return [d.reset_index(drop=True) for d in (train, val, test)]


class ReviewDataset(Dataset):
    """Tokenise -> truncate to max_length -> right-pad with token 50256.
    Returns (input_ids, true_length, label); true_length lets the model read the LAST REAL token."""
    def __init__(self, df, tokenizer, task, max_length=C.MAX_LENGTH, pad_id=C.PAD_ID):
        enc = [tokenizer.encode(t) for t in df["text"]]
        self.n_truncated = sum(len(e) > max_length for e in enc)
        enc = [e[:max_length] for e in enc]
        self.lengths = [len(e) for e in enc]
        width = max(self.lengths)
        self.inputs = [e + [pad_id] * (width - len(e)) for e in enc]
        self.labels = [C.to_label(r, task) for r in df["rating"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.tensor(self.inputs[i]), torch.tensor(self.lengths[i]),
                torch.tensor(self.labels[i]))


def make_loader(df, tok, task, batch_size=C.BATCH_SIZE, shuffle=False, seed=C.SEED):
    ds = ReviewDataset(df, tok, task)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=g), ds


def token_length_stats(df, tok):
    lens = [len(tok.encode(t)) for t in df["text"]]
    return {"mean": sum(lens) / len(lens), "min": min(lens), "max": max(lens),
            "pct_over_max_length": 100.0 * sum(l > C.MAX_LENGTH for l in lens) / len(lens)}, lens
