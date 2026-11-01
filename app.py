"""Gradio application.  Run: python app.py  (needs checkpoints/*.pt from run_experiments.py)"""
import os, torch, tiktoken, gradio as gr
import config as C, train_eval as T

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = tiktoken.get_encoding("gpt2")
models = {t: T.load_classifier(c["num_classes"], c["ckpt"], device)
          for t, c in C.TASKS.items() if os.path.exists(c["ckpt"])}
if not models:
    raise SystemExit("No checkpoint found - run `python run_experiments.py` first.")


def hotel_rating(review, task="5class"):
    """hotel_rating(review) -> predicted star rating (1-5) or 3-class group name."""
    task = task if task in models else next(iter(models))
    k, probs = T.classify(review, models[task], tok, device)
    if k is None:
        return None
    return k + 1 if task == "5class" else C.TASKS[task]["names"][k]


def ui(review, task):
    task = "5class" if task.startswith("5") else "3class"
    if task not in models:
        return "Model for this option has not been trained.", {}
    k, probs = T.classify(review, models[task], tok, device)
    if k is None:
        return "Please enter a review.", {}
    names = C.TASKS[task]["names"]
    label = f"{k + 1} star(s)" if task == "5class" else names[k]
    return f"Predicted: {label}", {n: p for n, p in zip(names, probs)}


demo = gr.Interface(
    fn=ui,
    inputs=[gr.Textbox(label="Hotel review", lines=6,
                       placeholder="The room was clean and the staff were very helpful..."),
            gr.Radio(["5-class (1-5 stars)", "3-class (neg / neutral / pos)"],
                     value="5-class (1-5 stars)", label="Model")],
    outputs=[gr.Textbox(label="Predicted rating"), gr.Label(label="Class probabilities")],
    title="Hotel Review Rating Predictor (GPT-2 124M, fine-tuned on TripAdvisor reviews)",
    description="Estimates a rating for a review imported without stars. "
                "It is an estimate to support, not replace, the reviewer's own rating.")

if __name__ == "__main__":
    demo.launch()
