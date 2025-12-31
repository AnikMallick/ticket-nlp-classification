from collections import Counter, defaultdict
from typing import Literal

import faiss
from sentence_transformers import SentenceTransformer

from ..data.rac_utils import retrive_related

def retrieval_model_predict(text: str, 
                            retrieval_model: SentenceTransformer,
                            index: faiss.Index,
                            corpus: dict,
                            keys: tuple[str], 
                            y_key: str,
                            k: int = 5,
                            remove_top: bool = False,
                            weight: Literal["count","score"] = "count") -> tuple[str, float]:
    
    _data = retrive_related(
        retrieval_model=retrieval_model,
        index=index,
        text=text,
        corpus=corpus,
        keys=keys,
        k=k,
        remove_top=remove_top
    )
    _r_y = [_r[y_key] for _r in _data]
    _r_s = [_r["score"] for _r in _data]
    if weight == "count":
        _counter = Counter()
        _counter.update(_r_y)
        y_pred, y_pred_count = _counter.most_common(1)[0]
        return y_pred, y_pred_count/float(len(_data))
    else:
        class_scores = defaultdict(float)
        for i, _y in enumerate(_r_y):
            class_scores[_y] += _r_s[i]
            
        predicted_label = max(class_scores, key=class_scores.get)
        total_score = sum(class_scores.values())
        weighted_score = class_scores[predicted_label] / total_score  if total_score > 0 else 0.0
        return predicted_label, float(weighted_score)