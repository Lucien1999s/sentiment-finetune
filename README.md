# sentiment-finetune

Finetune a 3-class sentiment model (negative / neutral / positive) end-to-end.
This repo demonstrates how to build a reliable NLP fine-tuning pipeline that you can clone, run, and reproduce with a single command.

---

## Quickstart

```bash
git clone git@github.com:Lucien1999s/sentiment-finetune.git
```

```bash
cd sentiment-finetune
```

```bash
pip install -r requirements.txt
```

### Run（`data/`、`model/` can be empty）

**Quick (smoke test)**

```bash
python train.py --quick
```

**Full training**

```bash
python train.py
```

**Inference**

```bash
python infer.py --text 'This place is awesome!' --model model/roberta_yelp3
```

**Gradio demo（optional）**

```bash
python app.py --model model/roberta_yelp3
```

> The training command will automatically: download & transform the dataset → split train/val/test → tokenize → train (with class weights) → evaluate → save model & metrics to `model/roberta_yelp3/`.

**Training Faster for Free**

You can run it on Google colab for free GPU, you have to change execution phase from CPU to GPU:

```bash
!git clone https://github.com/Lucien1999s/sentiment-finetune.git
```

```bash
%cd sentiment-finetune
```

```bash
!pip install -r requirements.txt
```

```bash
!python train.py --quick
```

---

## Demo

A minimal UI to try predictions interactively:

![Gradio demo](doc/gradio_ui.png)

---

## Technical Details

[![Alt text](doc/finetune_strategy.png)](doc/finetuning_guide.md)


---

## License

MIT (code). Dataset follows the original dataset’s license/terms.
