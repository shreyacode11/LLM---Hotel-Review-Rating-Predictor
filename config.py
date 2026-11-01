"""Shared configuration for the Hotel Review Rating Predictor mini project (MP-C35)."""
SEED = 123
CSV_PATH = "data/tripadvisor_hotel_reviews.csv"   # Kaggle file: columns Review, Rating
SUBSET_SIZE = 10_000                              # stratified subset of the ~20,000 reviews
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15                 # remaining 15 % = test
MAX_LENGTH = 256                                  # truncation length in tokens
PAD_ID = 50256                                    # <|endoftext|> used as padding token
BATCH_SIZE = 8
EPOCHS = 3
LR = 5e-5
WEIGHT_DECAY = 0.1
UNFREEZE_LAST_N = 1                               # last N transformer blocks are trainable
EVAL_FREQ = 50
EVAL_ITER = 10

GPT_CONFIG_124M = {
    "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.0, "qkv_bias": True,
}

TASKS = {
    "5class": {"num_classes": 5, "ckpt": "checkpoints/gpt2_hotel_5class.pt",
               "names": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]},
    "3class": {"num_classes": 3, "ckpt": "checkpoints/gpt2_hotel_3class.pt",
               "names": ["Negative (1-2)", "Neutral (3)", "Positive (4-5)"]},
}

def to_label(rating, task):
    """5class: rating 1..5 -> 0..4.  3class: 1-2 -> 0, 3 -> 1, 4-5 -> 2."""
    rating = int(rating)
    if task == "5class":
        return rating - 1
    return 0 if rating <= 2 else (1 if rating == 3 else 2)

FIVE_TO_THREE = [0, 0, 1, 2, 2]                   # maps 5-class index to 3-class index
