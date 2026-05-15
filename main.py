import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import src.config as cfg
from src.data_loader import generate_imbalanced_data, prepare_data_splits
from src.trainer import train_random_forest, train_pytorch_mlp
from src.evaluate import (
    get_predictions, calculate_metrics, plot_pr_curves_styled, plot_calibration_curves
)

def run_cross_validation():
    print(f"[1/4] Generating Imbalanced Dataset ({int(cfg.IMBALANCE_RATIO[0]*100)}% Major / {int(cfg.IMBALANCE_RATIO[1]*100)}% Minor)...")
    X, y = generate_imbalanced_data()
    
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    
    experiments = [
        ("Random Forest", "baseline", False),
        ("Random Forest", "class_weight", False),
        ("Random Forest", "smote", False),
        ("PyTorch MLP", "baseline", True),
        ("PyTorch MLP", "class_weight", True),
        ("PyTorch MLP", "smote", True),
        ("PyTorch MLP", "focal_loss", True),
    ]
    
    summary_results = []
    pr_curve_data = []
    calib_data = []

    print(f"[2/4] Running {cfg.N_FOLDS}-Fold Stratified Cross-Validation Benchmark...")
    
    for model_name, strat, is_pytorch in experiments:
        print(f"  -> Benchmarking: {model_name} [{strat}]...")
        
        fold_metrics = {"PR_AUC": [], "ROC_AUC": [], "F1_Macro": [], "Balanced_Acc": [], "MCC": []}
        

        first_fold_vis = None
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            

            X_train_s, X_val_s, y_train_s, y_val_s = prepare_data_splits(
                np.vstack((X_train, X_val)), np.hstack((y_train, y_val)), test_size=len(y_val)/len(y)
            )
            
            if is_pytorch:
                model = train_pytorch_mlp(X_train_s, y_train_s, strategy=strat)
            else:
                model = train_random_forest(X_train_s, y_train_s, strategy=strat)
                
            probs, preds = get_predictions(model, X_val_s, is_pytorch=is_pytorch)
            m, prec, rec = calculate_metrics(y_val_s, probs, preds)
            
            for k in fold_metrics:
                fold_metrics[k].append(m[k])
                
            if fold == 0:
                first_fold_vis = {"prec": prec, "rec": rec, "probs": probs, "y_true": y_val_s}

        # Format mean ± std
        res_row = {"Model": model_name, "Strategy": strat}
        for k, vals in fold_metrics.items():
            res_row[k] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        summary_results.append(res_row)
        
        pr_curve_data.append({
            "Model": model_name, 
            "Strategy": strat, 
            "Precision": first_fold_vis["prec"], 
            "Recall": first_fold_vis["rec"]
        })
        
        if strat in ["baseline", "smote"] and model_name == "Random Forest":
            calib_data.append({
                "Model": model_name,
                "Strategy": strat,
                "probs": first_fold_vis["probs"],
                "y_true": first_fold_vis["y_true"]
            })

    print("[3/4] Generating Benchmark Visualizations...")
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    plot_pr_curves_styled(pr_curve_data)
    plot_calibration_curves(calib_data)

    print("[4/4] Saving Results Table...")
    df_results = pd.DataFrame(summary_results)
    df_results.to_csv(cfg.RESULTS_CSV, index=False)

    print("\n================ STRATIFIED 5-FOLD CV BENCHMARK SUMMARY ================")
    print(df_results.to_string(index=False))
    print("=======================================================================")
    print(f"\nExecution complete. Artifacts saved to '{cfg.OUTPUT_DIR}/'.")

if __name__ == "__main__":
    run_cross_validation()