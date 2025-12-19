from functools import partial
from sklearn.metrics import f1_score
import torch

from ..evaluation.neural_eval import evaluate

from ..data.collate import create_batch

from ..data.datastruct import Batch, Sample
from ..models.base_neural_model import BaseModule
from tqdm import tqdm
from .earlystopping import EarlyStopping

def train_one_epoch(model: BaseModule, 
                    batchs: list[Batch],
                    criterion: torch.nn.CrossEntropyLoss,
                    optimizer: torch.optim.Optimizer):
    
    model.train()
    
    y_pred = [] 
    y_true = []
    total_loss = 0
    batch_count = 0

    for btach in tqdm(batchs):
        optimizer.zero_grad()
        
        logit = model(btach.input_ids, btach.attention_masks)
        loss = criterion(logit, btach.labels)
        
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item()
        batch_count += 1
        
        y_pred.extend(logit.argmax(1).cpu().numpy())
        y_true.extend(btach.labels.cpu().numpy())
    
    return total_loss / batch_count, y_pred, y_true

def train_model(model: BaseModule, total_epochs:int, train_samples: list[Sample], 
                training_batch_size: int, pad_id: int, device: str, 
                test_batches: list[Batch], criterion: torch.nn.CrossEntropyLoss,
                early_stop: bool = False, lr: float = 1e-4, random_state: int = 43):
    history = []
    
    create_batch_partial = partial(create_batch,
                               batch_size=training_batch_size,
                               pad_id=pad_id, 
                               device=device, shuffle=True, 
                               random_state=43)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stop_obj = EarlyStopping(mode='max')
    
    for epoch in tqdm(range(total_epochs)):
        _epoch_train_batch = create_batch_partial(in_samples=train_samples)
        train_loss, train_preds, train_labels = train_one_epoch(model=model, batchs=_epoch_train_batch, 
                                                                criterion=criterion, optimizer=optimizer)
        test_loss, test_preds, test_labels = evaluate(model, test_batches, criterion)
        
        train_f1 = f1_score(train_labels, train_preds, average="macro")
        test_f1 = f1_score(test_labels, test_preds, average="macro")
        print(f"""
        Epoch {epoch+1}
        Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}
        Val   Loss: {test_loss:.4f} | Val   F1: {test_f1:.4f}
        """)
        history.append((train_loss, test_loss, train_f1, test_f1))
        if early_stop and early_stop_obj.check(test_f1):
            print(f"Early stopping at epoch: {epoch}")
            return history
        
    return history