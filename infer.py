# infer.py
import os, sys, json, argparse, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_model(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval().to(get_device())
    return tok, mdl

def predict_texts(tokenizer, model, texts, max_len=256, return_probs=False):
    device = next(model.parameters()).device
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        if return_probs:
            probs = softmax(logits, dim=-1).cpu().numpy()
        else:
            probs = None
    labels = [ID2LABEL[int(i)] for i in preds]
    return labels, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="model/roberta_yelp3", help="訓練輸出的目錄")
    ap.add_argument("--text", help="單句推論：直接給一句英文")
    ap.add_argument("--file", help="批量推論：txt 檔案，一行一筆")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--probs", action="store_true", help="輸出每類機率")
    ap.add_argument("--json", action="store_true", help="以 JSON 格式輸出（方便程式處理）")
    args = ap.parse_args()

    if not (args.text or args.file):
        print("請用 --text 'your sentence' 或 --file path.txt 指定輸入")
        sys.exit(1)

    tokenizer, model = load_model(args.model_dir)

    if args.text:
        texts = [args.text]
    else:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [ln.strip() for ln in f if ln.strip()]

    labels, probs = predict_texts(tokenizer, model, texts, args.max_len, args.probs)

    if args.json:
        out = []
        for i, t in enumerate(texts):
            item = {"text": t, "label": labels[i]}
            if probs is not None:
                item["scores"] = {ID2LABEL[j]: float(probs[i][j]) for j in range(probs.shape[1])}
            out.append(item)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for i, t in enumerate(texts):
            if args.probs and probs is not None:
                scores = ", ".join([f"{ID2LABEL[j]}={probs[i][j]:.3f}" for j in range(probs.shape[1])])
                print(f"[{labels[i]}] {scores} | {t}")
            else:
                print(f"[{labels[i]}] {t}")

if __name__ == "__main__":
    # 在 Apple Silicon 上，遇到不支援的 op 會自動 fallback 到 CPU
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
