from dataclasses import dataclass
import torch

@dataclass
class Sample:
    input_ids: torch.Tensor
    label: torch.Tensor

@dataclass
class Batch:
    input_ids: torch.Tensor # shape: [B, T] batch size, max token_ids for the samples in the batch
    labels: torch.Tensor # shape: [B]
    attention_masks: torch.Tensor # shape: [B, T]
    