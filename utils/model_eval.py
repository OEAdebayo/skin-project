import numpy as np
from keras import models
from dataclasses import dataclass
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@dataclass(frozen=True)
class ModelPerformance:
    accuracy_score: float
    recall: float
    f1: float
    precision: float
    auc: float

def model_metric(model: models.Model, 
                 x_dat: np.ndarray, 
                 y_dat: np.ndarray,
                 classification_type: str) -> ModelPerformance:
    
    if classification_type not in ("mc", "bc"):
        raise ValueError(f"The 'classification_type' parameter only takes values 'mc' or 'bc' but classification_type: {classification_type} was given")

    if classification_type == "mc":
        y_pred_prob = model.predict(x_dat, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_dat = np.argmax(y_dat, axis=1)

        accuracy = accuracy_score(y_dat, y_pred)
        recall = recall_score(y_dat, y_pred, average='weighted')
        f1 = f1_score(y_dat, y_pred, average='weighted')
        precision = precision_score(y_dat, y_pred, average='weighted')

        y_dat_binarized = label_binarize(y_dat, classes=np.arange(y_pred_prob.shape[1]))
        auc = roc_auc_score(y_dat_binarized, y_pred_prob, multi_class='ovr')

    elif classification_type == "bc":
        y_pred_prob = model.predict(x_dat, verbose=0)
        y_pred = [1 if y > 0.5 else 0 for y in y_pred_prob]
        y_dat = y_dat.astype(int)

        accuracy = accuracy_score(y_dat, y_pred)
        precision = precision_score(y_dat, y_pred, average='binary')
        recall = recall_score(y_dat, y_pred, average='binary')
        f1 = f1_score(y_dat, y_pred, average='binary')
        auc = roc_auc_score(y_dat, y_pred_prob)

    return ModelPerformance(
        accuracy_score=accuracy,
        recall=recall,
        f1=f1,
        precision=precision,
        auc=auc
    )