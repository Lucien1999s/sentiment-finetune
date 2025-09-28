from pathlib import Path

BASE_MODEL   = "distilroberta-base"
DATASET_NAME = "yelp_review_full"

DATA_DIR  = Path("data/yelp3")
MODEL_DIR = Path("model/roberta_yelp3")

MAX_LEN   = 256
BATCH     = 32
LR        = 2e-5
EPOCHS    = 2
VAL_SPLIT = 0.1

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_CLASSES = 3
