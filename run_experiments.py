"""End-to-end pipeline: data -> BEFORE eval -> fine-tuning -> save/reload -> AFTER eval
-> figures -> confusing reviews -> 5-class vs 3-class comparison.
Usage: python run_experiments.py   (writes results/*.json, results/*.png, checkpoints/*.pt)"""
import json, os, random, time
import numpy as np, torch, tiktoken, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C, data_prep as D, train_eval as T


def plot_confusion(cm, names, title, path):
    cm = np.array(cm)
    plt.figure(figsize=(5, 4.2))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(names)), names, rotation=30, ha="right")
    plt.yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def main():
    os.makedirs("results", exist_ok=True)
    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = tiktoken.get_encoding("gpt2")

    df, stats = D.load_reviews()
    train_df, val_df, test_df = D.make_splits(df)
    tl, lens = D.token_length_stats(df, tok)
    stats.update(train=len(train_df), val=len(val_df), test=len(test_df), token_len=tl,
                 subset_rating_counts={int(k): int(v) for k, v in
                                       pd_counts(train_df, val_df, test_df).items()})
    plt.figure(); plt.hist(lens, bins=50); plt.axvline(C.MAX_LENGTH, color="r", ls="--")
    plt.xlabel("Tokens per review"); plt.ylabel("Count"); plt.title("Review length (tokens)")
    plt.savefig("results/fig2_1_token_hist.png", dpi=200, bbox_inches="tight"); plt.close()

    results = {"stats": stats, "device": device, "tasks": {}}
    for task, tc in C.TASKS.items():
        k, names = tc["num_classes"], tc["names"]
        print(f"\n######## TASK {task} ########")
        loaders = {s: D.make_loader(d, tok, task, shuffle=(s == "train"))
                   for s, d in (("train", train_df), ("val", val_df), ("test", test_df))}
        stats[f"truncated_{task}"] = {s: loaders[s][1].n_truncated for s in loaders}

        model = T.build_classifier(k, device)
        total, trainable = T.count_params(model)
        yp, yt, _ = T.predict_loader(model, loaders["test"][0], device)
        before = T.compute_metrics(yt, yp, k)
        print(f"BEFORE fine-tuning: acc {before['accuracy']:.4f}")

        t0 = time.time()
        hist = T.train_model(model, loaders["train"][0], loaders["val"][0], device,
                             C.EPOCHS, C.EVAL_FREQ, C.EVAL_ITER, tc["ckpt"])
        secs = time.time() - t0

        model = T.load_classifier(k, tc["ckpt"], device)          # verifies save + reload
        yp, yt, probs = T.predict_loader(model, loaders["test"][0], device)
        after = T.compute_metrics(yt, yp, k)
        print(f"AFTER fine-tuning: acc {after['accuracy']:.4f} f1 {after['macro_f1']:.4f} mae {after['mae']:.4f}")
        out = {"num_classes": k, "total_params": total, "trainable_params": trainable,
               "before": before, "after": after, "train_seconds": secs, "history": hist}
        if task == "5class":                                       # collapse to 3 groups for a fair comparison
            g = np.array(C.FIVE_TO_THREE)
            out["after_collapsed_to_3_accuracy"] = float((g[yp] == g[yt]).mean())

        plt.figure()
        plt.plot(hist["steps"], hist["train"], label="Training loss")
        plt.plot(hist["steps"], hist["val"], "--", label="Validation loss")
        plt.xlabel("Optimiser step"); plt.ylabel("Cross-entropy loss"); plt.legend()
        plt.title(f"Fine-tuning loss ({task})")
        plt.savefig(f"results/fig4_1_loss_{task}.png", dpi=200, bbox_inches="tight"); plt.close()
        plot_confusion(after["confusion_matrix"], names, f"Test confusion matrix ({task})",
                       f"results/fig4_2_confusion_{task}.png")

        wrong = np.where(yp != yt)[0]                              # most confident mistakes
        top = sorted(wrong, key=lambda i: -probs[i][yp[i]])[:5]
        out["confusing_reviews"] = [{"text": test_df["text"][int(i)], "true_rating": int(test_df["rating"][int(i)]),
                                     "true": names[yt[i]], "predicted": names[yp[i]],
                                     "confidence": float(probs[i][yp[i]])} for i in top]
        results["tasks"][task] = out

    json.dump(results, open("results/results.json", "w"), indent=1)
    print(json.dumps({t: {"before_acc": r["before"]["accuracy"], "after_acc": r["after"]["accuracy"],
                          "f1": r["after"]["macro_f1"], "mae": r["after"]["mae"]}
                      for t, r in results["tasks"].items()}, indent=1))


def pd_counts(*dfs):
    import pandas as pd
    return pd.concat(dfs)["rating"].value_counts().sort_index().to_dict()


if __name__ == "__main__":
    main()
