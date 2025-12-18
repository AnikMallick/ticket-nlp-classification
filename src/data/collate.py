import torch

from .datastruct import Sample, Batch

def collate_func(samples: list[Sample], pad_id: int) -> Batch:
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
    
    return Batch(
        input_ids=padded_inputs,
        labels=torch.stack(labels),
        attention_masks=attention_masks
    )