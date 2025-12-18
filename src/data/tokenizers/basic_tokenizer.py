import re
from collections import Counter
import json
import os
from .base import BaseTokenizer

class BasicWordTokenizer(BaseTokenizer):
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
    
    def _get_tokens(self, text: str) -> list[str]:
        return text.split(" ")
    
    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = text.strip()
        return text
    
    def fit(self, texts: list[str]) -> None:
        text_counter: Counter = Counter()
        
        for text in texts:
            _clean_text = self._clean_text(text)
            _tokens = self._get_tokens(_clean_text)
            text_counter.update(_tokens)
        
        capped_text_counter = text_counter.most_common(self.max_vocab_size)
        
        for id, (token, _) in enumerate(capped_text_counter, start=2):
            self.token_to_id[token] = id
            self.id_to_token[id] = token
        
        self.fitted = True
        print("Tokenizer fitted. Vocab size: ", len(self.token_to_id))
    
    def encode(self, text: str) -> list[int]:
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted.")

        _clean_text = self._clean_text(text)
        _tokens = self._get_tokens(_clean_text)
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unknown_key]) for token in _tokens]
        
        return token_ids
    
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
