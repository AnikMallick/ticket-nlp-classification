from collections import Counter, defaultdict
from functools import partial
from typing import Callable, Literal
from sklearn.feature_extraction.text import TfidfVectorizer


import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .rac_utils import retrive_related, retrive_related_embedding

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import faiss


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
                            remove_top: bool = False) -> list[float]:
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
            features.append(float(class_scores_weighted[_c]))
        else:
            features.append(0.0)
    
    # vote entropy
    _entropy = normalized_entropy(features[:len(classes)], len(classes))
    features.append(_entropy)
    
    # top-1 and margin/ambiguity similarity 
    classes_sorted = [k for k, v in sorted(class_scores_weighted.items(), key=lambda x: x[1], reverse=True)]
    features.append(float(class_scores_weighted[classes_sorted[0]]))
    if len(classes_sorted) >= 2:
        features.append(float(class_scores_weighted[classes_sorted[0]] - class_scores_weighted[classes_sorted[1]]))
    else:
        features.append(1.0) # so only one class and giving 1.0 so the margin is the max
    
    return features


def get_retrieval_features_data(data: dict, 
                            y_key: str,
                            classes: list[str],) -> list[float]:
    features = []
    
    # class probabilities 
    class_scores = defaultdict(float)
    for i, _d in enumerate(data):
        class_scores[_d[y_key]] += _d["score"]
    total_score = sum(class_scores.values())
    weight_score = lambda val: val / total_score  if total_score > 0 else 0.0
    class_scores_weighted = {k: weight_score(v) for k, v in class_scores.items()}
    for _c in classes:
        if _c in class_scores_weighted:
            features.append(float(class_scores_weighted[_c]))
        else:
            features.append(0.0)
    
    # vote entropy
    _entropy = normalized_entropy(features[:len(classes)], len(classes))
    features.append(_entropy)
    
    # top-1 and margin/ambiguity similarity 
    classes_sorted = [k for k, v in sorted(class_scores_weighted.items(), key=lambda x: x[1], reverse=True)]
    # features.append(float(class_scores_weighted[classes_sorted[0]]))
    if len(classes_sorted) >= 2:
        features.append(float(class_scores_weighted[classes_sorted[0]] - class_scores_weighted[classes_sorted[1]]))
    else:
        features.append(1.0) # so only one class and giving 1.0 so the margin is the max
    
    return features

def argument_data(texts: list[str],
                vectorizer: TfidfVectorizer, 
                retrieval_model: SentenceTransformer,
                index: faiss.Index,
                corpus: dict,
                keys: tuple[str], 
                y_key: str,
                classes: list[str],
                k: int = 5,
                remove_top: bool = False):
    _retrieval_features_0 = get_retrieval_features(
            text=texts[0],
            retrieval_model=retrieval_model,
            index=index,
            corpus=corpus,
            keys=keys,
            y_key=y_key,
            classes=classes,
            k=k,
            remove_top=remove_top
        )
    _tfidf_vector_0 = vectorizer.transform([texts[0]]).toarray()[0]
    _augmented_data = np.hstack([_tfidf_vector_0, _retrieval_features_0])
    
    for i in range(1, len(texts)):
        _retrieval_features_i = get_retrieval_features(
            text=texts[i],
            retrieval_model=retrieval_model,
            index=index,
            corpus=corpus,
            keys=keys,
            y_key=y_key,
            classes=classes,
            k=k,
            remove_top=remove_top
        )
        _tfidf_vector_i = vectorizer.transform([texts[i]]).toarray()[0]
        _augmented_data_i = np.hstack([_tfidf_vector_i, _retrieval_features_i])
        _augmented_data  = np.vstack([_augmented_data, _augmented_data_i.reshape(1, -1)])
    
    _augmented_data


async def task_retrieval_features(embedding, 
               index: faiss.Index,
                corpus: dict,
                keys: tuple[str],
                y_key: str,
                classes: list[str],
                k: int = 5,
                remove_top: bool = False):
    return get_retrieval_features_data(retrive_related_embedding(index=index, 
                                       embedding=embedding, 
                                       corpus=corpus, keys=keys,
                                       k=k,
                                       remove_top=remove_top),
                                y_key=y_key,
                                classes=classes)

async def agument_data_optimized(texts: list[str],
                vectorizer: TfidfVectorizer, 
                retrieval_model: SentenceTransformer,
                index_path: str,
                index_count: int,
                corpus: dict,
                keys: tuple[str], 
                y_key: str,
                classes: list[str],
                k: int = 5,
                remove_top: bool = False):
    
    embeddings = retrieval_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True).astype("float32")
    vectors = vectorizer.transform(texts)
    
    indexes = [faiss.read_index(index_path) for _ in range(index_count)]

    _retrieved_features = []
    for i in tqdm(range(0, embeddings.shape[0], index_count)):
        _coros = [
            task_retrieval_features(
                embedding=embeddings[j].reshape(1, -1), 
                index=indexes[j - i], corpus=corpus, 
                keys=keys, k=k, remove_top=remove_top,
                y_key=y_key, classes=classes
            )
            for j in range(i, min(len(texts), i + index_count))
        ]
        results = await asyncio.gather(*_coros)
        _retrieved_features.extend(results)
    _retrieved_features = np.asarray(_retrieved_features)
    return np.hstack([vectors.toarray(), _retrieved_features])


