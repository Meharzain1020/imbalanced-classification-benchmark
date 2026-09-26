<img width="1376" height="768" alt="mage" src="https://github.com/user-attachments/assets/399f8056-4179-418a-885a-cd34893ec6c9" />

A statistical machine learning benchmark evaluating class-imbalance strategies (97:3 ratio) across **Random Forest** and **PyTorch MLP** architectures using **Stratified 5-Fold Cross-Validation**. 

This repository explores the trade-offs between decision-threshold metrics (F1-Macro, MCC) and ranking-based probability calibration (PR-AUC), providing empirical insights relevant to real-world applications such as **rare disease diagnosis, credit fraud scoring, structural anomaly detection, and satellite land-cover mapping**.

---

## Technical Insights & Core Findings

### The Rebalancing Paradox
When handling severe class imbalance, standard interventions like **SMOTE** (synthetic oversampling) or cost-sensitive **Class Weighting** are typically applied to boost minority class detection. This benchmark demonstrates a crucial trade-off:

* **Threshold Decision Accuracy:** Rebalancing significantly improves binary classification metrics at default decision boundaries ($p = 0.5$), yielding higher **F1-Macro** and **Matthews Correlation Coefficient (MCC)** scores.
* **Probability Calibration Distortion:** Rebalancing artificially inflates minority class distributions during training. As shown in calibration curve analysis, this forces the model into overconfident predictions, degrading its overall **Precision-Recall AUC (PR-AUC)**.
* **Engineering Impact:** In practical deployment scenarios where outputs serve as continuous risk scores (e.g., medical triage prioritization or automated defect ranking), **baseline models optimized via post-hoc probability calibration or threshold tuning** outperform synthetic oversampling methods.

---

## Strategy & Model Architecture Overview

### Evaluated Strategies
* **Baseline:** Standard empirical risk minimization (default loss and sampling).
* **Class Weighting:** Cost-sensitive learning via class-proportional loss scaling.
* **SMOTE:** Synthetic Minority Over-sampling Technique applied dynamically to training folds.
* **Focal Loss:** Dynamic loss adaptation ($\alpha = 0.25, \gamma = 2.0$) prioritizing hard negative instances in deep networks.

### Model Architectures
* **Random Forest:** Ensemble of 100 decision trees (`scikit-learn`).
* **PyTorch MLP:** Multi-layer perceptron with Batch Normalization, ReLU activations, and Dropout (`torch.nn`).

---

<img width="2400" height="1800" alt="calibration_curves" src="https://github.com/user-attachments/assets/bebd7ee5-0ed6-43d8-aebd-b8c8ee16100a" />
<img width="2700" height="1800" alt="pr_curves" src="https://github.com/user-attachments/assets/4616df17-7c85-4314-bd05-5215816a8989" />


## Domain Relevance & Mitacs Project Alignment

This project models complex data distributions common in computational science and machine learning research:

* **Medical AI & Clinical Diagnostics:** Evaluates model reliability when detecting rare pathological anomalies without inducing high false-positive alarm rates.
* **Earth Observation & Remote Sensing:** Outlines techniques for identifying low-frequency surface features (e.g., wildfire origins, standing water cover) in high-dimensional raster arrays.
* **Predictive Maintenance & Quality Control:** Demonstrates robust evaluation setups for monitoring structural defects or sensor degradation where failure events are rare.
* **Financial Data Analytics:** Benchmark framework directly applies to high-volume transaction scoring and anomaly detection.

---

## Repository Structure

```
imbalanced-classification-benchmark/
├── data/                      # Runtime dataset directory
├── notebook/
│   └── demo.ipynb             # Interactive walkthrough notebook
├── outputs/
│   ├── figures/               # Generated PR & Calibration plots
│   └── benchmark_results.csv  # Final metrics summary table (Mean ± Std)
├── src/
│   ├── __init__.py
│   ├── config.py              # Central experiment parameters & random seeds
│   ├── data_loader.py         # Stratified data generation & standardization
│   ├── models.py              # PyTorch MLP & custom Focal Loss definitions
│   ├── trainer.py             # SKLearn & PyTorch cross-validation loops
│   └── evaluate.py            # PR-AUC, MCC, and calibration visualizers
├── tests/
│   ├── test_data_loader.py    # Pytest validation for array shapes & ratios
│   ├── test_models.py         # Tensor dimension and loss function unit tests
│   └── test_evaluate.py       # Metric range and calculation unit tests
├── main.py                    # Stratified 5-Fold Cross-Validation driver
├── requirements.txt
├── .gitignore
└── README.md

```

## How to Use

1. Install dependencies from `requirements.txt` by writing in terminal `pip install -r requirements.txt`.
2. Run `python main.py` to optimize input and generate reports
3. Check `outputs/figures/` for visualizations 

---


## Author

**Zia Ur Rehman**

**AI Engineer**
