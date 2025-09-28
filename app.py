import argparse
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentiment_finetune.config import ID2LABEL

def launch(model_path_or_id: str):
    tok = AutoTokenizer.from_pretrained(model_path_or_id, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path_or_id)
    mdl.eval()

    def infer_one(text: str):
        with torch.no_grad():
            enc = tok([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
            logits = mdl(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            top = int(probs.argmax())
        return ID2LABEL[top], {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}

    gr.Interface(
        fn=infer_one,
        inputs=gr.Textbox(label="Input"),
        outputs=[gr.Label(label="Prediction"), gr.Label(label="Probabilities")],
        title="Sentiment (3-class) – Finetuned",
    ).launch()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="model/roberta_yelp3")
    args = ap.parse_args()
    launch(args.model)

if __name__ == "__main__":
    main()
