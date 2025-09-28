from typing import Optional
import torch
import torch.nn.functional as F
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: Optional[list] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss
