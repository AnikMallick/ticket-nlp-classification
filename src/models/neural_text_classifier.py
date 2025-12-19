import torch
import torch.nn as nn
from ..data.datastruct import Batch
from .base_neural_model import BaseModule

class TicketTextClassifierV01(BaseModule):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_id: int, n_classes: int):
        super().__init__()
        
        self.hidden_layer1 = 256
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_layer1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_layer1, n_classes)
        )
    
    def forward(self, inputs, attention_masks):
        # B batch size, T max token_ids for the samples in the batch, D embedding_dim
        embeddings = self.embedding(inputs) # shape: [B, T, D]
        
        mask = attention_masks.unsqueeze(-1) # shape: [B, T, 1]
        masked_embeddings = embeddings * mask # embedding for all the padids go to zero
        
        summed = masked_embeddings.sum(dim=1) # shape: [B, D] summed all the embediings for all the token ids
        # to normalize we will devide by the count of token_ids non pad
        count = mask.sum(dim=1).clamp(min=1e-6) # shape: [B, 1], clamp is used to we dont get 0 as count and avoide devide by zero
        
        pooled = summed / count # shape: [B, D]
        
        logit = self.classifier(pooled) # in shape: [B, D], out shape: [B, n_classes]
        return logit