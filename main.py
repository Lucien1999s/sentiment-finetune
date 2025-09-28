import os, json, random, numpy as np
from collections import Counter
from typing import Dict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict, load_from_disk, ClassLabel
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments
)
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

# ========= Config =========
BASE_MODEL  = os.getenv("BASE_MODEL", "distilroberta-base")      # 英文任務穩定好用
DATASET_NAME = os.getenv("DATASET_NAME", "yelp_review_full")
SEED       = int(os.getenv("SEED", 42))
MAX_LEN    = int(os.getenv("MAX_LEN", 256))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
LR         = float(os.getenv("LR", 2e-5))
EPOCHS     = int(os.getenv("EPOCHS", 2))
VAL_SPLIT  = float(os.getenv("VAL_SPLIT", 0.1))
SMALL_TRAIN = int(os.getenv("SMALL_TRAIN", "0"))  # e.g. 2100；0 表示不用
SMALL_TEST  = int(os.getenv("SMALL_TEST", "0"))   # e.g. 2000；0 表示不用
SMALL_STRATIFIED = os.getenv("SMALL_STRATIFIED", "1") in ("1","true","yes")

# 路徑：資料與模型都放在 repo root 下
DATA_DIR  = os.getenv("DATA_DIR",  "data/yelp3")
MODEL_DIR = os.getenv("MODEL_DIR", "model/roberta_yelp3")

# 類別權重：auto / none
CLASS_WEIGHTS_MODE = os.getenv("CLASS_WEIGHTS", "auto").lower()  # "auto" 或 "none"

# 自動判斷 fp16（無 CUDA 就關閉；MPS 也關閉）
USE_FP16 = os.getenv("FP16", "").lower() in ["1", "true", "yes"] or (
    os.getenv("FP16", "") == "" and "CUDA_VISIBLE_DEVICES" in os.environ
)

id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {v: k for k, v in id2label.items()}


# ========= Utils =========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

def map_star_to_3class(example):
    # yelp_review_full: label ∈ {0,1,2,3,4} → 星=label+1
    star = int(example["label"]) + 1
    if star <= 2:
        y = 0  # negative
    elif star == 3:
        y = 1  # neutral
    else:
        y = 2  # positive
    return {"labels": y, "text": example["text"]}  # HF Trainer 需要欄位名 'labels'

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }

def stratified_small(dataset, per_class: int, seed: int) -> 'datasets.Dataset':
    """從 Dataset 依 labels 分層等量抽樣。要求 labels 是 ClassLabel/int。"""
    by_label = {i: [] for i in range(3)}
    for i, y in enumerate(dataset["labels"]):
        by_label[int(y)].append(i)
    rng = np.random.default_rng(seed)
    idxs = []
    for y in range(3):
        take = min(per_class, len(by_label[y]))
        idxs.extend(rng.choice(by_label[y], size=take, replace=False).tolist())
    rng.shuffle(idxs)
    return dataset.select(idxs)


# ========= Load & Prep =========
def build_yelp_3class() -> DatasetDict:
    raw = load_dataset(DATASET_NAME)

    train = raw["train"].map(map_star_to_3class, remove_columns=raw["train"].column_names)
    test  = raw["test"].map(map_star_to_3class,  remove_columns=raw["test"].column_names)

    # labels 轉 ClassLabel，才能 stratify
    class_names = ["negative", "neutral", "positive"]
    label_feature = ClassLabel(names=class_names)
    train = train.cast_column("labels", label_feature)
    test  = test.cast_column("labels",  label_feature)

    # （可選）縮小資料量做 smoke test
    if SMALL_TRAIN and SMALL_TRAIN > 0:
        if SMALL_STRATIFIED:
            per_class = max(1, SMALL_TRAIN // 3)
            train = stratified_small(train, per_class=per_class, seed=SEED)
        else:
            train = train.shuffle(seed=SEED).select(range(min(SMALL_TRAIN, len(train))))
    if SMALL_TEST and SMALL_TEST > 0:
        test = test.shuffle(seed=SEED).select(range(min(SMALL_TEST, len(test))))

    # 分層切 validation
    split = train.train_test_split(test_size=VAL_SPLIT, seed=SEED, stratify_by_column="labels")
    ds = DatasetDict(train=split["train"], validation=split["test"], test=test)
    return ds

def load_or_build_dataset() -> DatasetDict:
    """優先從 data/ 載入；若無則建立並保存到 data/。"""
    if os.path.isdir(DATA_DIR):
        print(f"[Data] Loading dataset from disk: {DATA_DIR}")
        return load_from_disk(DATA_DIR)
    print(f"[Data] Building dataset from '{DATASET_NAME}' and saving to {DATA_DIR} ...")
    ds = build_yelp_3class()
    Path(DATA_DIR).parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(DATA_DIR)
    print("[Data] Saved.")
    return ds

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)


# ========= Weighted Trainer (class-imbalance friendly) =========
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    # 兼容 transformers 新舊版本：多接 num_items_in_batch，且保留 return_outputs
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,   # ★ 新增這個參數以相容新版本
    ):
        labels = inputs.get("labels")
        # 移除 labels 避免傳兩份
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ========= Main =========
def main():
    set_seed(SEED)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    ds = load_or_build_dataset()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer),
                       batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3, id2label=id2label, label2id=label2id
    )

    # 類別權重（用 train 分布計算）
    cnt = Counter(tokenized["train"]["labels"])
    total = sum(cnt.values())
    num_classes = 3
    class_weights = None
    if CLASS_WEIGHTS_MODE == "auto":
        class_weights = [total / (num_classes * max(1, cnt[i])) for i in range(num_classes)]
        print("[Class Weights]", class_weights)
    else:
        print("[Class Weights] none")

    # 嘗試用新參數；若不支援則退回舊參數（相容舊版 transformers）
    try:
        args = TrainingArguments(
            output_dir=MODEL_DIR,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=LR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
            fp16=USE_FP16,
        )
    except TypeError:
        # 舊版 fallback
        print("[Info] Using legacy TrainingArguments (eval_steps/save_steps).")
        args = TrainingArguments(
            output_dir=MODEL_DIR,
            learning_rate=LR,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            logging_steps=50,
            logging_first_step=True,
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            fp16=USE_FP16,
        )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Start training...")
    trainer.train()

    print("Evaluate on validation:")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    print("Evaluate on test:")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(test_metrics)

    preds = trainer.predict(tokenized["test"])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=-1)

    print("\nClassification report (test):")
    print(classification_report(
        y_true, y_pred, target_names=[id2label[i] for i in range(3)], digits=4))

    # 儲存最佳模型（新參數會自動回到 best；舊參數則保存最後一個）
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\n[Model] Saved to: {MODEL_DIR}")

    # 保存 metrics 與 混淆矩陣到檔案
    out = {
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "class_distribution_train": {id2label[i]: int(cnt[i]) for i in range(3)},
        "class_weights": class_weights,
    }
    with open(Path(MODEL_DIR) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    np.savetxt(Path(MODEL_DIR) / "confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")
    # 盡量不強依賴繪圖；若你要圖，再自行 pip install matplotlib 並打開下列代碼
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        ticks = [id2label[i] for i in range(3)]
        plt.xticks(range(3), ticks, rotation=45)
        plt.yticks(range(3), ticks)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        fig.savefig(Path(MODEL_DIR) / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        print("[Warn] matplotlib not available or plot failed:", e)

if __name__ == "__main__":
    main()
