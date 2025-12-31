from typing import Literal
import faiss
from sentence_transformers import SentenceTransformer
from ..data.tokenizers.base import BaseTokenizer
from .base_neural_model import BaseModule
from .retrieval_model import retrieval_model_predict
from ..evaluation.neural_eval import inference_one

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

class ConfidenceAwareHybridModel:
    def __init__(self,
                 model: BaseModule,
                 tokenizer: BaseTokenizer,
                 classes: list[str],
                 device: str,
                 retrieval_model: SentenceTransformer,
                 index: faiss.Index,
                 corpus: dict,
                 keys: tuple[str],
                 y_key: str, 
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.classes = classes
        self.device = device
        self.retrieval_model = retrieval_model
        self.index = index
        self.corpus = corpus
        self.keys = keys
        self.y_key = y_key
    
    def predict(self, text: str, threshold: float, k: int, remove_top: bool = False, 
                retrieval_weight: Literal["count","score"] = "count") -> tuple[str, float, Literal["r", "nn"]]:
        _y_pred_r, _y_pred_r_score = retrieval_model_predict(
                                        text=text,
                                        retrieval_model=self.retrieval_model,
                                        index=self.index,
                                        corpus=self.corpus,
                                        keys=self.keys,
                                        y_key=self.y_key,
                                        k=k,
                                        remove_top=remove_top,
                                        weight=retrieval_weight
                                        )
        if _y_pred_r_score > threshold:
            return _y_pred_r, _y_pred_r_score, "r"
        else:
            _y_pred_n, _y_pred_n_prob, _all_probs =  inference_one(
                text,
                model=self.model,
                tokenizer=self.tokenizer, 
                classes=self.classes, 
                device=self.device
            )
            return _y_pred_n, _y_pred_n_prob, "nn"


class ConfidenceAwareHybridModelSklearn:
    def __init__(self,
                 model: LinearSVC | LogisticRegression | xgb.XGBClassifier,
                 vectorizer: TfidfVectorizer,
                 classes: list[str],
                 retrieval_model: SentenceTransformer,
                 index: faiss.Index,
                 corpus: dict,
                 keys: tuple[str],
                 y_key: str, 
                 ):
        self.model = model
        self.vectorizer = vectorizer
        self.classes = classes
        self.retrieval_model = retrieval_model
        self.index = index
        self.corpus = corpus
        self.keys = keys
        self.y_key = y_key
        self.is_xgboost = isinstance(model, xgb.XGBClassifier)
    
    def predict(self, text: str, threshold: float, k: int, remove_top: bool = False, 
                retrieval_weight: Literal["count","score"] = "count") -> tuple[str, float, Literal["r", "sk"]]:
        _y_pred_r, _y_pred_r_score = retrieval_model_predict(
                                        text=text,
                                        retrieval_model=self.retrieval_model,
                                        index=self.index,
                                        corpus=self.corpus,
                                        keys=self.keys,
                                        y_key=self.y_key,
                                        k=k,
                                        remove_top=remove_top,
                                        weight=retrieval_weight
                                        )
        if _y_pred_r_score > threshold:
            return _y_pred_r, _y_pred_r_score, "r"
        else:
            _X_vec = self.vectorizer.transform([text])
            _y_pred_s = self.model.predict(_X_vec)
            if self.is_xgboost:
                return self.classes[int(list(_y_pred_s)[0])], -1, "sk"
            return list(_y_pred_s)[0], -1, "sk"