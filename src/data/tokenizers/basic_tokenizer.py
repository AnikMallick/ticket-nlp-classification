import re
from collections import Counter
import json
import os
from typing import Callable, Literal, Optional
from .base import BaseTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed

class BasicWordTokenizer(BaseTokenizer):
    def _get_tokens(self, text: str) -> list[str]:
        return text.split(" ")
    
    def _get_tokens_bi(self, text: str, uni_tokens: Optional[list[str]] = None) -> list[str]:
        if not uni_tokens:
            uni_tokens = self._get_tokens(text)
        bi_tokens = []
        for i in range(0, len(uni_tokens) -1):
            bi_tokens.append(f"{uni_tokens[i]}_{uni_tokens[i+1]}")
        return bi_tokens
        
    
    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = text.strip()
        return text
    
    def fit(self, texts: list[str], n_gram: int | tuple[int, int] = 1) -> None:
        if self.fitted:
            raise RuntimeError("Tokenizer already fitted.")
        
        if n_gram not in {1, 2, (1, 2)}:
            raise ValueError("Unigram, Bigram and Unigram+Bigram only supported.")

        self.n_gram_mode = n_gram
        text_counter: Counter = Counter()
        
        for text in texts:
            _clean_text = self._clean_text(text)
            _tokens = []
            match n_gram:
                case 1:
                    _tokens.extend(self._get_tokens(_clean_text))
                case 2:
                    _tokens.extend(self._get_tokens_bi(_clean_text))
                case (1, 2):
                    _uni_tokens = self._get_tokens(_clean_text)
                    _tokens.extend(_uni_tokens)
                    _tokens.extend(self._get_tokens_bi(_clean_text, _uni_tokens))

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
        _tokens = []
        match self.n_gram_mode:
            case 1:
                _tokens.extend(self._get_tokens(_clean_text))
            case 2:
                _tokens.extend(self._get_tokens_bi(_clean_text))
            case (1, 2):
                _uni_tokens = self._get_tokens(_clean_text)
                _tokens.extend(_uni_tokens)
                _tokens.extend(self._get_tokens_bi(_clean_text, _uni_tokens))
        
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unknown_key]) for token in _tokens]
        
        return token_ids
    
class BasicCharTokenizer(BaseTokenizer):
    def _get_words(self, text: str) -> list[str]:
        return text.split(" ")
    
    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = text.strip()
        return text
    
    def _get_tokens(self, word: str, n_gram: int) -> list[str | None]:
        if len(word) < n_gram:
            return []
        elif len(word) == n_gram:
            return [word]
        return [word[i: i+n_gram] for i in range(0, len(word) - n_gram + 1)]
    
    def _get_tokens_from_words(self, words: list[str], n_gram: int) -> list[str | None]:
        _tokens = []
        for word in words:
            _tokens.extend(self._get_tokens(word, n_gram=n_gram))
        return _tokens
    
    def _get_tokens_3_task(self, sent: str) -> list[str | None]:
        _clean_sent = self._clean_text(sent)
        _clean_words = self._get_words(_clean_sent)
        return self._get_tokens_from_words(_clean_words, n_gram=3)
    
    def _get_tokens_4_task(self, sent: str) -> list[str | None]:
        _clean_sent = self._clean_text(sent)
        _clean_words = self._get_words(_clean_sent)
        return self._get_tokens_from_words(_clean_words, n_gram=4)
    
    def _get_tokens_5_task(self, sent: str) -> list[str | None]:
        _clean_sent = self._clean_text(sent)
        _clean_words = self._get_words(_clean_sent)
        return self._get_tokens_from_words(_clean_words, n_gram=5)
    
    def _update_text_counter(self, task: Callable[[str], list[str | None]],
                             texts: list[str],
                             counter_to_update: Counter) -> None:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(task, text) for text in texts]
            
            for f in as_completed(futures):
                try:
                    counter_to_update.update(f.result())
                except Exception as e:
                    print("Task failed:", e)
    
    def fit(self, texts: list[str], n_gram: int | tuple[int, int] = (3, 5)) -> None:
        if self.fitted:
            raise RuntimeError("Tokenizer already fitted.")
        
        if n_gram not in {3, 4, 5, (3, 4), (4, 5), (3, 5)}:
            raise ValueError("Not supported n-gram. 3, 4, 5, (3, 4), (4, 5), (3, 5)")

        self.n_gram_mode = n_gram
        text_counter: Counter = Counter()
        match n_gram:
            case 3:
                self._update_text_counter(self._get_tokens_3_task, texts, text_counter)
            case 4:
                self._update_text_counter(self._get_tokens_4_task, texts, text_counter)
            case 5:
                self._update_text_counter(self._get_tokens_5_task, texts, text_counter)
            case (3, 4):
                self._update_text_counter(self._get_tokens_3_task, texts, text_counter)
                self._update_text_counter(self._get_tokens_4_task, texts, text_counter)
            case (4, 5):
                self._update_text_counter(self._get_tokens_4_task, texts, text_counter)
                self._update_text_counter(self._get_tokens_5_task, texts, text_counter)
            case (3, 5):
                self._update_text_counter(self._get_tokens_3_task, texts, text_counter)
                self._update_text_counter(self._get_tokens_4_task, texts, text_counter)
                self._update_text_counter(self._get_tokens_5_task, texts, text_counter)
                        
        capped_text_counter = text_counter.most_common(self.max_vocab_size)
        
        for id, (token, _) in enumerate(capped_text_counter, start=2):
            self.token_to_id[token] = id
            self.id_to_token[id] = token
        
        self.fitted = True
        print("Tokenizer fitted. Vocab size: ", len(self.token_to_id))
    
    def encode(self, text: str) -> list[int]:
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted.")

        _tokens = []
        match self.n_gram_mode:
            case 3:
                _tokens.extend(self._get_tokens_3_task(text))
            case 4:
                _tokens.extend(self._get_tokens_4_task(text))
            case 5:
                _tokens.extend(self._get_tokens_5_task(text))
            case (3, 4):
                _tokens.extend(self._get_tokens_3_task(text))
                _tokens.extend(self._get_tokens_4_task(text))
            case (4, 5):
                _tokens.extend(self._get_tokens_4_task(text))
                _tokens.extend(self._get_tokens_5_task(text))
            case (3, 5):
                _tokens.extend(self._get_tokens_3_task(text))
                _tokens.extend(self._get_tokens_4_task(text))
                _tokens.extend(self._get_tokens_5_task(text))
        
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unknown_key]) for token in _tokens]
        
        return token_ids
    
class CustomWordCharTokenizer(BaseTokenizer):
    def _get_words(self, text: str) -> list[str]:
        return text.split(" ")
    
    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = text.strip()
        return text
    
    def _get_tokens(self, word: str, n_gram: int) -> list[str | None]:
        if len(word) < n_gram:
            return []
        elif len(word) == n_gram:
            return [word]
        return [word[i: i+n_gram] for i in range(0, len(word) - n_gram + 1)]
    
    def _get_tokens_from_words(self, words: list[str], n_gram: int) -> list[str | None]:
        _tokens = []
        for word in words:
            _tokens.extend(self._get_tokens(word, n_gram=n_gram))
        return _tokens
    
    def _get_tokens_3_task(self, sent: str) -> list[str | None]:
        _clean_sent = self._clean_text(sent)
        _clean_words = self._get_words(_clean_sent)
        return self._get_tokens_from_words(_clean_words, n_gram=3)
    
    def _get_tokens_4_task(self, sent: str) -> list[str | None]:
        _clean_sent = self._clean_text(sent)
        _clean_words = self._get_words(_clean_sent)
        return self._get_tokens_from_words(_clean_words, n_gram=4)
    
    def _get_tokens_5_task(self, sent: str) -> list[str | None]:
        _clean_sent = self._clean_text(sent)
        _clean_words = self._get_words(_clean_sent)
        return self._get_tokens_from_words(_clean_words, n_gram=5)
    
    def _update_text_counter(self, task: Callable[[str], list[str | None]],
                             texts: list[str],
                             counter_to_update: Counter) -> None:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(task, text) for text in texts]
            
            for f in as_completed(futures):
                try:
                    counter_to_update.update(f.result())
                except Exception as e:
                    print("Task failed:", e)
    
    def fit(self, texts: list[str], n_gram: int | tuple[int, int] = None) -> None:
        if self.fitted:
            raise RuntimeError("Tokenizer already fitted.")
        
        if n_gram :
            raise Warning("n_gram is not used in custom tokenizer, base config is word unigram + char-gram-(3, 5)")

        # self.n_gram_mode = n_gram
        text_counter: Counter = Counter()
        self._update_text_counter(self._get_tokens_3_task, texts, text_counter)
        self._update_text_counter(self._get_tokens_4_task, texts, text_counter)
        self._update_text_counter(self._get_tokens_5_task, texts, text_counter)
        for text in texts:
            _clean_sent = self._clean_text(text)
            _clean_words = self._get_words(_clean_sent)
            text_counter.update(_clean_words)
            
        capped_text_counter = text_counter.most_common(self.max_vocab_size)
        
        for id, (token, _) in enumerate(capped_text_counter, start=2):
            self.token_to_id[token] = id
            self.id_to_token[id] = token
        
        self.fitted = True
        print("Tokenizer fitted. Vocab size: ", len(self.token_to_id))
    
    def encode(self, text: str) -> list[int]:
        if not self.fitted:
            raise RuntimeError("Tokenizer not fitted.")
        
        _tokens = []
        _tokens.extend(self._get_tokens_3_task(text))
        _tokens.extend(self._get_tokens_4_task(text))
        _tokens.extend(self._get_tokens_5_task(text))
        _clean_sent = self._clean_text(text)
        _clean_words = self._get_words(_clean_sent)
        _tokens.extend(_clean_words)
        
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unknown_key]) for token in _tokens]
        
        return token_ids


