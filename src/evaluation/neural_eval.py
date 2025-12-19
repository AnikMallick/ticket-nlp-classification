import torch
import numpy as np

from ..data.tokenizers.base import BaseTokenizer

from ..data.datastruct import Batch
from ..models.base_neural_model import BaseModule

@torch.no_grad()
def evaluate(model: BaseModule, 
             batches: list[Batch],
             criterion: torch.nn.CrossEntropyLoss):
    model.eval()
    
    y_pred = [] # ignore: Type
    y_true = [] # ignore: Type
    total_loss = 0
    batch_count = 0
    for btach in batches:
        logits = model(btach.input_ids, btach.attention_masks)
        loss = criterion(logits, btach.labels)
        
        total_loss += loss.item()
        batch_count += 1

        y_pred.extend(logits.argmax(1).cpu().numpy())
        y_true.extend(btach.labels.cpu().numpy())
        
    return total_loss / batch_count, y_pred, y_true

@torch.no_grad()
def inference_one(text: str, model: BaseModule , tokenizer: BaseTokenizer, classes: list[str],
                  device: str):
    model.eval()
    tokens = tokenizer.encode(text)
    attention_mask = [1.0] * len(tokens)
    tokens_padded = tokenizer.pad(tokens)
    attention_mask = attention_mask + [0.0] * (len(tokens_padded) - len(tokens))
    tokens_padded_tensor = torch.from_numpy(np.array(tokens_padded, dtype=np.long)).unsqueeze(0).to(device)
    attention_mask_tensor = torch.from_numpy(np.array(attention_mask, dtype=np.long)).unsqueeze(0).to(device)
    with torch.inference_mode():
        result_logit = model(tokens_padded_tensor, attention_mask_tensor)
    class_pred = classes[result_logit.argmax(1).item()]
    classes_probabilities = torch.nn.functional.softmax(result_logit, dim=1)
    return class_pred, classes_probabilities.max(1).values.item(), classes_probabilities


