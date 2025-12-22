from sentence_transformers import SentenceTransformer
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def retrive_related(retrieval_model: SentenceTransformer,
                    index: faiss.Index,
                    text: str,
                    corpus: dict,
                    keys: tuple[str], 
                    k: int = 5,
                    remove_top: bool = False) -> list[str]:
    top_k = k if not remove_top else k+1
    
    _emb = retrieval_model.encode([text], normalize_embeddings=True).astype("float32")
    _related_scores, _related_indexes = index.search(_emb, top_k)
    
    retrieved = []
    for index, score in zip(_related_indexes[0], _related_scores[0]):
        if remove_top and index == 0:
            continue
        
        _data = {
            key: corpus[key][index] for key in keys
        }
        _data["score"] = score
        retrieved.append(_data)
    return retrieved


def agument_text(retrieval_model: SentenceTransformer, 
                index: faiss.Index,
                text: str,
                corpus: dict,
                keys: tuple[str],
                augmentation_key: str,
                k: int = 5,
                remove_top: bool = False,
                augmentation_format: str = "{text} [CONTEXT] {contexts}") -> str:
    _related = retrive_related(
        retrieval_model, 
        index,
        text,
        corpus,
        keys, 
        k,
        remove_top,
    )
    
    _augmentation_text = [_r[augmentation_key] for _r in _related]
    
    return augmentation_format.format(
        text=text,
        contexts=" ".join(_augmentation_text)
    )

def pooled_agument_texts(retrieval_model: SentenceTransformer, 
                index: faiss.Index,
                texts: list[str],
                corpus: dict,
                keys: tuple[str],
                augmentation_key: str,
                k: int = 5,
                remove_top: bool = False,
                augmentation_format: str = "{text} [CONTEXT] {contexts}") -> list[str]:
    task = lambda text: agument_text(
        retrieval_model=retrieval_model, 
        index=index,
        corpus=corpus,
        keys=keys,
        augmentation_key=augmentation_key,
        k=k,
        remove_top=remove_top,
        augmentation_format=augmentation_format,
        text=text,
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
    
        futures = [executor.submit(task, text) for text in texts]
        results = []
        for f in tqdm(as_completed(futures)):
            try:
                results.append(f.result())
            except Exception as e:
                print("Task failed:", e)
        return results
    