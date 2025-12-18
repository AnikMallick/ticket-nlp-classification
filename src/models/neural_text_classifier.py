import torch
import torch.nn as nn
from ..data.datastruct import Batch

class NeuralTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_id: int, n_classes: int):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)
    
    def forward(self, batch: Batch):
        # B batch size, T max token_ids for the samples in the batch, D embedding_dim
        embeddings = self.embedding(batch.input_ids) # shape: [B, T, D]
        
        mask = batch.attention_masks.unsqueeze(-1) # shape: [B, T, 1]
        masked_embeddings = embeddings * mask # embedding for all the padids go to zero
        
        summed = masked_embeddings.sum(dim=1) # shape: [B, D] summed all the embediings for all the token ids
        # to normalize we will devide by the count of token_ids non pad
        count = mask.sum(dim=1).clamp(min=1e-6) # shape: [B, 1], clamp is used to we dont get 0 as count and avoide devide by zero
        
        pooled = summed / count # shape: [B, D]
        
        logit = self.classifier(pooled) # in shape: [B, D], out shape: [B, n_classes]
        return logit