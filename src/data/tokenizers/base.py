import json
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    def __init__(self, max_vocab_size: int = 30000, max_length: int = 256):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        
        self.pad_key = "<PAD>"
        self.unknown_key = "<UNK>"
        
        self.token_to_id: dict[str,int] = {
            self.pad_key: 0,
            self.unknown_key: 1
        }
        self.id_to_token: dict[int, str] = {
            v: k for k, v in self.token_to_id.items()
        }
        
        self.fitted = False
        self.n_gram_mode = 1
    
    @abstractmethod    
    def fit(self, texts: list[str], n_gram) -> None: 
        pass
    
    @abstractmethod  
    def encode(self, text: str) -> list[int]:
        pass
    
    def pad(self, token_ids: list[int]) -> list[int]:
        pad_length = self.max_length - len(token_ids)
        return token_ids + [self.token_to_id[self.pad_key]] * max(0, pad_length)
    
    def encode_pad(self, texts: str) -> list[int]:
        return self.pad(self.encode(texts))
    
    def vocab_dump(self, fp) -> None:
        json.dump(self.token_to_id, fp)
    
    def load_vocab(self, vocab: dict[str, int]) -> None:
        if self.fitted:
            raise RuntimeError("Tokenizer already fitted.")
        if len(vocab) > self.max_vocab_size + 2:
            raise ValueError("Given vocab size gratter than allowed size.")
        
        for token, id in vocab.items():
            self.token_to_id = vocab
            self.id_to_token = {
                v: k for k, v in self.token_to_id.items()
            }
    
    def get_pad_id(self) -> int:
        return self.token_to_id[self.pad_key]
    
    def get_unknown_id(self) -> int:
        return self.token_to_id[self.unknown_key]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def is_fitted(self) -> bool:
        return self.fitted
