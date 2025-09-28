import argparse
from sentiment_finetune.infer_core import predict_texts
from sentiment_finetune.config import ID2LABEL

def build_parser():
    p = argparse.ArgumentParser(prog="infer", description="CLI inference for 3-class sentiment.")
    p.add_argument("--text", type=str, help="single text")
    p.add_argument("--file", type=str, help="path to a file with one text per line")
    p.add_argument("--model", type=str, default="model/roberta_yelp3")
    return p

def main():
    args = build_parser().parse_args()
    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts += [line.strip() for line in f if line.strip()]
    if not texts:
        print("No input provided. Use --text or --file.")
        return
    preds, probs = predict_texts(texts, args.model)
    for t, p, pr in zip(texts, preds, probs):
        print(f"[{ID2LABEL[int(p)]}] {t}\n  prob={pr}")

if __name__ == "__main__":
    main()
