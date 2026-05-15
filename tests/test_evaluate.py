import numpy as np
from src.evaluate import calculate_metrics

def test_metrics_range():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.15, 0.05, 0.2, 0.3, 0.1, 0.4, 0.8, 0.9])
    preds = (probs >= 0.5).astype(int)
    
    metrics, prec, rec = calculate_metrics(y_true, probs, preds)
    
    assert 0.0 <= metrics["PR_AUC"] <= 1.0
    assert 0.0 <= metrics["ROC_AUC"] <= 1.0
    assert 0.0 <= metrics["F1_Macro"] <= 1.0
    assert 0.0 <= metrics["Balanced_Acc"] <= 1.0
    assert -1.0 <= metrics["MCC"] <= 1.0