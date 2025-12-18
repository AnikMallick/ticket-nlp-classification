import torch
from torch.utils.data import Dataset
from .tokenizers.base import BaseTokenizer
from .datastruct import Sample


class TicketDataset(Dataset):
    def __init__(self, texts: list[str], labels: list, tokenizer: BaseTokenizer):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, index) -> Sample:
        token_ids = self.tokenizer.encode(self.texts[index])
        
        return Sample(
            input_ids=torch.tensor(token_ids, dtype=torch.long),
            label=torch.tensor(self.labels[index], dtype=torch.long)
        )