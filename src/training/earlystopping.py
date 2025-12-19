from typing import Literal

class EarlyStopping:
    def __init__(self, patience: int = 3, mode: Literal["max", "min"] = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_score = None
        self.not_improved_counter = 0
        self.early_stop_flag = False
    
    def check(self, current_score) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        current_improvement = (
            current_score - self.best_score
            if self.mode == "max"
            else self.best_score - current_score
        )
        
        if current_improvement > self.min_delta:
            self.best_score = current_score
            self.not_improved_counter = 0
        else:
            self.not_improved_counter += 1
        
        if self.not_improved_counter >= self.patience:
            self.early_stop_flag = True
            return True
        return False
            