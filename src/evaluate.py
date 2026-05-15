import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, balanced_accuracy_score, 
    roc_auc_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve
import src.config as cfg

def get_predictions(model, X_test, is_pytorch=False):
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32).to(cfg.DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
    else:
        probs = model.predict_proba(X_test)[:, 1]
    
    preds = (probs >= 0.5).astype(int)
    return probs, preds

def calculate_metrics(y_true, probs, preds):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, probs)
    f1_macro = f1_score(y_true, preds, average="macro")
    bal_acc = balanced_accuracy_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    
    return {
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "F1_Macro": f1_macro,
        "Balanced_Acc": bal_acc,
        "MCC": mcc
    }, precision, recall

def plot_pr_curves_styled(pr_curve_data, output_path=f"{cfg.FIGURES_DIR}/pr_curves.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 6))
    
    # Visual Encoding: Color = Model Type, Linestyle = Strategy
    color_map = {"Random Forest": "#1f77b4", "PyTorch MLP": "#ff7f0e"}
    style_map = {"baseline": "-", "class_weight": "--", "smote": "-.", "focal_loss": ":"}
    
    for item in pr_curve_data:
        m_name = item["Model"]
        strat = item["Strategy"]
        c = color_map.get(m_name, "black")
        ls = style_map.get(strat, "-")
        
        label = f"{m_name} ({strat})"
        plt.plot(item['Recall'], item['Precision'], label=label, color=c, linestyle=ls, linewidth=2)
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves Across Models and Strategies")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_calibration_curves(calib_data, output_path=f"{cfg.FIGURES_DIR}/calibration_curves.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    
    for item in calib_data:
        prob_true, prob_pred = calibration_curve(item['y_true'], item['probs'], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f"{item['Model']} ({item['Strategy']})")
        
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (True Frequency)")
    plt.title("Probability Calibration Distortion from Rebalancing")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()