import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_texts(texts, model_id_or_path):
    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    mdl.eval()
    with torch.no_grad():
        enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        logits = mdl(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)
    return preds, probs
