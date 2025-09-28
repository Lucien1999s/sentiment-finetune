Awesome idea — putting the deeper write-up under `doc/` keeps the README lean.
Below is a complete, polished **English** guide you can save as `doc/finetuning_guide.md`.

---

# Finetuning Guide — sentiment-finetune

This document explains **what** the project does and, more importantly, **why** it is designed this way. It walks from raw data to a reproducible three-class sentiment model, covering data schema, label mapping, tokenization, loss design, training loop, evaluation, and portability.

---

## 1. Problem framing

We finetune an encoder-based transformer (default: `distilroberta-base`) for **three-class sentiment analysis** on English reviews:

* `0 = negative`, `1 = neutral`, `2 = positive`.

Why three classes? In many real applications, “neutral” is operationally meaningful (e.g., customer support triage). Two-class setups often collapse uncertain or mixed opinions into an arbitrary side; a dedicated neutral class forces the model to learn the middle region.

---

## 2. Dataset and label mapping

**Source.** We use Hugging Face’s `yelp_review_full`, which contains review **text** and integer **label** ∈ {0,…,4}, corresponding to **stars** 1–5 (star = label+1).

**Mapping 5→3.**

* stars 1–2 → `negative (0)`
* star 3 → `neutral (1)`
* stars 4–5 → `positive (2)`

Rationale:

* Yelp’s 3-star reviews are typically “meh / acceptable,” a good proxy for neutral.
* Class boundaries align with human expectations and preserve headroom for the model to represent uncertainty.

**Implementation.** See `sentiment_finetune/data.py → map_star_to_3class`. After mapping, we **cast** the column to a `ClassLabel` with names `["negative","neutral","positive"]`. This improves type-safety and plays nicely with stratified operations.

**Stratified splits and quick mode.**
We build a `DatasetDict(train/validation/test)` with a stratified `train/validation` split on `labels`. For fast iteration, the CLI supports a **quick mode** that draws stratified subsamples (`train≈2100`, `test≈2000`) to keep class ratios stable even in small runs.

**Caching.**
On first run, the processed dataset is saved to `data/yelp3/` (Arrow). Subsequent runs reuse it to avoid repeated downloads and preprocessing.

---

## 3. Data schema after preprocessing

Each split (HF `Dataset`) contains:

* `text: str`
* `labels: ClassLabel(3)` with values `{0,1,2}`

Operations downstream (tokenization, batching, Trainer) consume these columns directly.

---

## 4. Tokenization and model head

**Tokenizer.** We use `AutoTokenizer` from the chosen backbone with:

* `truncation=True, padding=True, max_length=256`.

Length 256 balances coverage and speed/VRAM for typical reviews. For longer domains (support tickets, news), consider 384–512 with a smaller batch.

**Backbone and head.**
`AutoModelForSequenceClassification(num_labels=3, id2label, label2id)` adds a linear classification head on top of the pooled representation. The project defaults to `distilroberta-base` as a strong/compact English baseline; you can switch to `roberta-base` for higher capacity if your GPU allows.

---

## 5. Loss design and class imbalance

Sentiment datasets often skew toward positives. A neutral class is especially at risk of being under-represented. Using plain cross-entropy can bias the model toward majority classes.

**Class-weighted cross-entropy.**
We compute per-class weights from the training distribution:

[
w_c = \frac{\text{total}}{\text{num_classes} \times \max(1,\text{count}_c)}
]

and pass them to `torch.nn.functional.cross_entropy(logits, labels, weight=...)`.

**Where this lives.**
`sentiment_finetune/trainer.py` defines `WeightedTrainer`, which overrides `compute_loss` to inject class weights when `--class-weights auto` (default). Set `--class-weights none` to disable.

Why this approach?

* Minimal change to the training loop.
* Improves macro-F1 by preventing “neutral collapse.”
* Keeps probabilistic calibration reasonable (vs. aggressive resampling).

---

## 6. Training loop (HF Trainer)

We use `transformers.Trainer` for robust, cross-version training with a small compatibility shim.

**Core arguments (see `pipeline.py`).**

* Scheduling/eval: `evaluation_strategy="epoch"`, `save_strategy="epoch"`, `save_total_limit=2`.
* Model selection: `metric_for_best_model="macro_f1"`, `load_best_model_at_end=True`, `greater_is_better=True`.
* Knobs: `per_device_train_batch_size=32`, `per_device_eval_batch_size=32`, `learning_rate=2e-5`, `num_train_epochs=2`.
* Logging: `logging_steps=50`, `report_to="none"`.
* Precision: `fp16=True` automatically on **CUDA**; macOS **MPS** and CPU effectively run fp32 (safe). You can force `--no-fp16`.

**OS portability details.**

* On macOS/MPS, PyTorch warns that `pin_memory` is not supported. To silence: add `dataloader_pin_memory=False` in `TrainingArguments`.
* On Windows with some Python setups, many worker processes can hang; try `dataloader_num_workers=0–2`.
* We wrap `TrainingArguments` creation in a `try/except` to fall back for older `transformers` versions.

**Under the hood (optimization).**

* Optimizer defaults to **AdamW**, scheduler defaults to a linear decay.
* Training steps: forward pass → weighted CE loss → backprop via autograd → parameter update → LR step.

---

## 7. Evaluation and what to watch

**Metrics.**

* **Accuracy**: intuitive but can be misleading under imbalance.
* **Macro-F1**: plain average of class-wise F1; treats each class equally → our **selection metric**.
* **Weighted-F1**: frequency-weighted average; reflects overall user mix.

**Confusion matrix.**
We export a CSV (and PNG if `matplotlib` is present). Inspect the **neutral row/column** specifically—consistent off-diagonal mass indicates ambiguity between neutral and neighbors, guiding future data curation or thresholding strategies.

**Artifacts.**

* `model/roberta_yelp3/` contains best weights (`model.safetensors`), `config.json`, tokenizer files, plus `metrics.json` and `confusion_matrix.csv/png`.
* The repo ignores heavy artifacts by default but **keeps** metrics and the matrix so reviewers can validate results without cloning large files.

---

## 8. Command-line interface (training & inference)

* **Training** (builds the dataset if missing, then finetunes, evaluates, and saves):

  ```bash
  # quick smoke test (stratified subsample)
  python train.py --quick

  # full run (default knobs: epochs=2, batch=32, lr=2e-5, max_len=256)
  python train.py

  # examples
  python train.py --base-model roberta-base --epochs 3 --batch 32
  python train.py --class-weights none
  ```

* **Inference** (local path or HF Hub ID):

  ```bash
  python infer.py --text "This place is awesome!" --model model/roberta_yelp3
  # or file mode (one text per line)
  python infer.py --file path/to/texts.txt --model model/roberta_yelp3
  ```

* **Gradio demo** (optional):

  ```bash
  python app.py --model model/roberta_yelp3
  ```

---

## 9. Design choices and trade-offs

* **Three classes vs. two.**
  The neutral class carries product value and changes model behavior—macro-F1 becomes meaningful; class weighting becomes helpful.

* **DistilRoBERTa default.**
  Strong English coverage, fast on CPU/MPS, and small enough for constrained GPUs. If you have capacity, `roberta-base` often lifts F1 a bit further.

* **Weighted CE vs. resampling.**
  We prefer a loss-level fix first—it is simple, stable, and preserves the original data distribution, avoiding potential biases from oversampling.

* **Trainer over custom loop.**
  The HF Trainer gives mature checkpointing, logging, schedules, and evaluation hooks and is widely understood by reviewers. The small compatibility wrapper keeps it running across versions.

* **Stratified quick mode.**
  Developer experience matters. A one-flag subsample that preserves label ratios enables rapid iteration without silently changing class balance.

---

## 10. Reproducibility and portability

* **Determinism.** We fix seeds (`42`) for Python, NumPy, and PyTorch in `pipeline.py`. Complete determinism still depends on backend kernels and hardware, but runs are stable enough for iterative work.

* **Caching.** Processed datasets are stored under `data/yelp3/`. We never commit Arrow files to Git; they re-build deterministically if needed.

* **Cross-OS behavior.**

  * macOS (MPS): safe and convenient for prototyping; add `dataloader_pin_memory=False` to suppress the warning.
  * Linux (CUDA): fastest path; FP16 mixed precision enables larger effective batch sizes.
  * Windows: works; reduce `dataloader_num_workers` if you see IPC quirks.

---

## 11. Tuning guide (practical defaults)

* **Laptop (CPU/MPS):** start with `--quick`, then `--epochs 2`, `--batch 16–32`, `max_len=256`.
* **Cloud GPU (CUDA):** try `--batch 64`, `--epochs 3`; increase model to `roberta-base` if VRAM allows.
* **Learning rate:** default `2e-5`. If loss is jittery, try `1e-5`; if underfitting, try `3e-5`.
* **Further tricks:** label smoothing (`label_smoothing_factor=0.1`), early stopping via `TrainerCallback`, BF16 on A100/RTX 40xx (`bf16=True`), gradient accumulation to emulate larger batches.

---

## 12. Troubleshooting (quick)

* **Pin-memory warning on MPS.** Add `dataloader_pin_memory=False`. Harmless otherwise.
* **Neutral class underperforms.** Keep `--class-weights auto`; consider longer training or more neutral examples.
* **Windows dataloader stalls.** Use `dataloader_num_workers=0–2`.
* **OOM on GPU.** Lower `--batch`; optionally increase `--max-len` only if needed; use gradient accumulation if you must keep batch semantics.

---

## 13. Code map (where to look)

* **`sentiment_finetune/data.py`** — dataset loading, mapping 5→3, stratified splitting, caching.
* **`sentiment_finetune/config.py`** — defaults and label dicts.
* **`sentiment_finetune/trainer.py`** — `WeightedTrainer` with class-weighted CE.
* **`sentiment_finetune/metrics.py`** — accuracy, macro-F1, weighted-F1 for evaluation.
* **`sentiment_finetune/pipeline.py`** — end-to-end build → tokenize → train → evaluate → save.
* **`train.py` / `infer.py` / `app.py`** — thin CLIs over the library code.

---

## 14. Minimal conceptual refresher

1. **Tokenization.** Split text into subwords (BPE/WordPiece) → integer `input_ids`.
2. **Forward pass.** Encoder produces hidden states; classification head outputs `logits ∈ ℝ^3`.
3. **Loss.** Cross-entropy (with class weights) compares logits vs. true label.
4. **Backprop.** Autograd computes gradients; AdamW updates parameters; scheduler decays LR.
5. **Generalization.** Monitor macro-F1 and confusion matrix; avoid overfitting via early stopping or regularization if necessary.

---

### Appendix: High-level flow (ASCII)

```
Data (HF yelp_review_full)
   └─ map stars 1-2→neg, 3→neu, 4-5→pos
      └─ stratified train/val/test → cache(data/yelp3)
          └─ tokenize (fast, max_len=256)
              └─ model (distilroberta-base + CLS head)
                  └─ train (weighted CE, epochs, lr, batch)
                      ├─ eval per epoch (accuracy, macro-F1, weighted-F1)
                      ├─ select best by macro-F1
                      └─ save (weights+tokenizer, metrics.json, cm.csv/png)
```

If you want this guide split into multiple short pages (Data, Model, Training, Evaluation), say the word and I’ll refactor it into `doc/` sections with a mini table of contents.
