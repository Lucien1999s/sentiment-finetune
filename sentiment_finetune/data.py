from typing import Dict
from pathlib import Path
import numpy as np
from datasets import load_dataset, DatasetDict, load_from_disk, ClassLabel
from .config import DATASET_NAME, DATA_DIR, VAL_SPLIT, NUM_CLASSES, ID2LABEL

def map_star_to_3class(example: Dict):
    star = int(example["label"]) + 1
    if star <= 2: y = 0
    elif star == 3: y = 1
    else: y = 2
    return {"labels": y, "text": example["text"]}

def _stratified_small(dataset, per_class: int, seed: int):
    by = {i: [] for i in range(NUM_CLASSES)}
    for i, y in enumerate(dataset["labels"]): by[int(y)].append(i)
    rng = np.random.default_rng(seed)
    idxs = []
    for y in range(NUM_CLASSES):
        take = min(per_class, len(by[y]))
        idxs.extend(rng.choice(by[y], size=take, replace=False).tolist())
    rng.shuffle(idxs)
    return dataset.select(idxs)

def build_or_load_dataset(
    data_dir: Path = DATA_DIR,
    dataset_name: str = DATASET_NAME,
    seed: int = 42,
    small_train: int = 0,
    small_test: int = 0,
    stratified: bool = True,
) -> DatasetDict:
    if data_dir.is_dir():
        return load_from_disk(str(data_dir))

    raw = load_dataset(dataset_name)
    train = raw["train"].map(map_star_to_3class, remove_columns=raw["train"].column_names)
    test  = raw["test"].map(map_star_to_3class,  remove_columns=raw["test"].column_names)

    label_feature = ClassLabel(names=[ID2LABEL[i] for i in range(NUM_CLASSES)])
    train = train.cast_column("labels", label_feature)
    test  = test.cast_column("labels",  label_feature)

    if small_train and small_train > 0:
        if stratified:
            train = _stratified_small(train, per_class=max(1, small_train // NUM_CLASSES), seed=seed)
        else:
            train = train.shuffle(seed=seed).select(range(min(small_train, len(train))))
    if small_test and small_test > 0:
        test = test.shuffle(seed=seed).select(range(min(small_test, len(test))))

    split = train.train_test_split(test_size=VAL_SPLIT, seed=seed, stratify_by_column="labels")
    ds = DatasetDict(train=split["train"], validation=split["test"], test=test)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(data_dir))
    return ds
