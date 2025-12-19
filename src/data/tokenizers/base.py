class BaseTokenizer:
        
    def fit(self, texts: list[str]) -> None:
        pass
    
    def encode(self, text: str) -> list[int]:
        pass
    
    def pad(self, token_ids: list[int]) -> list[int]:
        pass
    
    def encode_pad(self, texts: str) -> list[int]:
        pass
    
    def vocab_dump(self, fp) -> None:
        pass
    
    def load_vocab(self, vocab: dict[str, int]) -> None:
        pass
    
    def get_pad_id(self) -> int:
        pass
    
    def get_unknown_id(self) -> int:
        pass

    def __len__(self) -> int:
        pass

    def is_fitted(self) -> bool:
        pass