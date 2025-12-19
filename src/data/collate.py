from typing import Optional
import torch
from copy import deepcopy

from .datastruct import Sample, Batch
import random
from itertools import batched
from functools import partial
from .tokenizers.base import BaseTokenizer


def create_samples(X: list[str], y: list, tokenizer: BaseTokenizer):
    samples = []
    for _X, _y in zip(X, y):
        samples.append(
            Sample(
                input_ids=torch.tensor(tokenizer.encode(_X), dtype=torch.long),
                label=torch.tensor(_y, dtype=torch.long)
            )
        )
    return samples

def create_batch(batch_size: int, in_samples: list[Sample], pad_id: int, device: Optional[str] = None,
                 shuffle: bool = False, random_state: int = 42) -> list[Batch]:
    samples = deepcopy(in_samples)
    if shuffle:
        random.seed(random_state)
        random.shuffle(samples)
    collate_func_partial = partial(collate_func, pad_id=pad_id, device=device)
    batchs = [collate_func_partial(samples=list(b_samples)) for b_samples in batched(samples, batch_size)]
    return batchs
    

def collate_func(samples: list[Sample], 
                 pad_id: int, device: Optional[str] = None) -> Batch:
    batch_input_ids = []
    labels = []
    max_length = 0
    batch_size = len(samples)
    for sample in samples:
        batch_input_ids.append(sample.input_ids)
        labels.append(sample.label)
        max_length = max(max_length, len(sample.input_ids))

    padded_inputs = torch.full(
        (batch_size, max_length),
        fill_value=pad_id,
        dtype=torch.long
    ) # init padded full tensor
    attention_masks = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long
    ) # init attention mask
    
    for i, sample_input_ids in enumerate(batch_input_ids):
        padded_inputs[i, :len(sample_input_ids)] = sample_input_ids
        attention_masks[i, :len(sample_input_ids)] = 1.0
    # now padded_inputs = actual data + pad_id
    # attention_max = 1.0 for all index where there is actual data + 0.0 for all the padded indexes
    
    if not device:
        return Batch(
            input_ids=padded_inputs,
            labels=torch.stack(labels),
            attention_masks=attention_masks
        )
    
    return Batch(
            input_ids=padded_inputs.to(device),
            labels=torch.stack(labels).to(device),
            attention_masks=attention_masks.to(device)
        )