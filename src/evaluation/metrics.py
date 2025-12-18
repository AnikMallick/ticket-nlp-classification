from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate(y_true, y_pred):
    print(classification_report(y_true, y_pred, digits=4))
    return confusion_matrix(y_true, y_pred)

def format_cm(cm, *, 
              class_names: list[str] = [], 
              normalize: bool = False) -> pd.DataFrame:
    if not class_names:
        class_names = [f"class-{i}" for i in range(len(cm))]
    
    cm_df = pd.DataFrame(cm,
                         index=[f"True: {c}" for c in class_names],
                         columns=[f"Pred: {c}" for c in class_names]
                         )
    if not normalize:
        return cm_df
    
    cm_normalized = cm_df.div(cm_df.sum(axis=1), axis=0)
    cm_normalized.round(3)
    return cm_normalized