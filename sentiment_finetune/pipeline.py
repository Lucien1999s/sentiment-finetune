import json, random
from collections import Counter
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments
)
from sklearn.metrics import classification_report, confusion_matrix

from .config import *
from .data import build_or_load_dataset
from .metrics import compute_metrics
from .trainer import WeightedTrainer

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def run_train(
    base_model: str = BASE_MODEL,
    dataset_name: str = DATASET_NAME,
    data_dir: str | Path = DATA_DIR,
    model_dir: str | Path = MODEL_DIR,
    seed: int = 42,
    max_len: int = MAX_LEN,
    batch: int = BATCH,
    lr: float = LR,
    epochs: int = EPOCHS,
    small_train: int = 0,
    small_test: int = 0,
    class_weights: str = "auto",
    fp16: Optional[bool] = None,
):
    data_dir = Path(data_dir); model_dir = Path(model_dir)
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)

    ds = build_or_load_dataset(data_dir, dataset_name, seed, small_train, small_test, stratified=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenized = ds.map(lambda x: tokenizer(x["text"], truncation=True, max_length=max_len),
                       batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=NUM_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID
    )

    cnt = Counter(tokenized["train"]["labels"])
    cw = None
    if class_weights == "auto":
        total = sum(cnt.values())
        cw = [total / (NUM_CLASSES * max(1, cnt[i])) for i in range(NUM_CLASSES)]

    if fp16 is None:
        fp16 = torch.cuda.is_available()

    try:
        args = TrainingArguments(
            output_dir=str(model_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="none",
            fp16=fp16,
            dataloader_pin_memory=False,
        )
    except TypeError:
        args = TrainingArguments(
            output_dir=str(model_dir),
            learning_rate=lr,
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=50,
            eval_steps=500,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            fp16=fp16,
        )

    trainer = WeightedTrainer(
        class_weights=cw,
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics  = trainer.evaluate()
    test_metrics = trainer.evaluate(tokenized["test"])

    preds  = trainer.predict(tokenized["test"])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=-1)

    print("\nClassification report (test):")
    print(classification_report(y_true, y_pred,
          target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)], digits=4))

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    out = {
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "class_distribution_train": {ID2LABEL[i]: int(cnt[i]) for i in range(NUM_CLASSES)},
        "class_weights": cw,
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    np.savetxt(model_dir / "confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")

    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        import matplotlib.pyplot as plt
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        ticks = [ID2LABEL[i] for i in range(NUM_CLASSES)]
        plt.xticks(range(NUM_CLASSES), ticks, rotation=45)
        plt.yticks(range(NUM_CLASSES), ticks)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
        fig.savefig(model_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        print("[Warn] matplotlib not available or plot failed:", e)
