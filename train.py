import argparse
from sentiment_finetune.pipeline import run_train

def build_parser():
    p = argparse.ArgumentParser(prog="train", description="Finetune 3-class sentiment model end-to-end.")
    p.add_argument("--base-model", default="distilroberta-base")
    p.add_argument("--dataset", default="yelp_review_full")
    p.add_argument("--data-dir", default="data/yelp3")
    p.add_argument("--model-dir", default="model/roberta_yelp3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--small-train", type=int, default=0, help="0 for full; >0 to subsample")
    p.add_argument("--small-test", type=int, default=0)
    p.add_argument("--class-weights", choices=["auto", "none"], default="auto")
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16 even if CUDA is available")
    p.add_argument("--quick", action="store_true", help="Shortcut: small-train=2100, small-test=2000")
    return p

def main():
    args = build_parser().parse_args()
    small_train = 2100 if args.quick else args.small_train
    small_test  = 2000 if args.quick else args.small_test
    run_train(
        base_model=args.base_model,
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        max_len=args.max_len,
        batch=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        small_train=small_train,
        small_test=small_test,
        class_weights=args.class_weights,
        fp16=False if args.no_fp16 else None,
    )

if __name__ == "__main__":
    main()
