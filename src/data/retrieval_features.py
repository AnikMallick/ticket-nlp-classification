from collections import Counter, defaultdict
from typing import Literal

import faiss
from sentence_transformers import SentenceTransformer
from .rac_utils import retrive_related

import numpy as np

def vote_entropy(vote_probs, eps=1e-12):
    p = np.array(vote_probs)
    p = p[p > 0]              # avoid log(0)
    return -np.sum(p * np.log(p + eps))

def normalized_entropy(vote_probs, num_classes):
    H = vote_entropy(vote_probs)
    return float(H / np.log(num_classes))

def get_retrieval_features(text: str, 
                            retrieval_model: SentenceTransformer,
                            index: faiss.Index,
                            corpus: dict,
                            keys: tuple[str], 
                            y_key: str,
                            classes: list[str],
                            k: int = 5,
                            remove_top: bool = False):
    features = []
    
    _data = retrive_related(
        retrieval_model=retrieval_model,
        index=index,
        text=text,
        corpus=corpus,
        keys=keys,
        k=k,
        remove_top=remove_top
    )
    
    # class probabilities 
    class_scores = defaultdict(float)
    for i, _d in enumerate(_data):
        class_scores[_d[y_key]] += _d["score"]
    total_score = sum(class_scores.values())
    weight_score = lambda val: val / total_score  if total_score > 0 else 0.0
    class_scores_weighted = {k: weight_score(v) for k, v in class_scores.items()}
    for _c in classes:
        if _c in class_scores_weighted:
            features.append(class_scores_weighted[_c])
        else:
            features.append(0.0)
    
    # vote entropy
    _entropy = normalized_entropy(features[:len(classes)], len(classes))
    features.append(_entropy)
    
    # top-1 and margin/ambiguity similarity 
    classes_sorted = [k for k, v in sorted(class_scores_weighted.items(), key=lambda x: x[1], reverse=True)]
    features.append(class_scores_weighted[classes_sorted[0]])
    if len(classes_sorted) >= 2:
        features.append(class_scores_weighted[classes_sorted[0]] - class_scores_weighted[classes_sorted[1]])
    else:
        features.append(1.0) # so only one class and giving 1.0 so the margin is the max
    
    return features
    