import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import src.config as cfg

def generate_imbalanced_data(
    n_samples=cfg.N_SAMPLES, 
    weights=cfg.IMBALANCE_RATIO, 
    random_state=cfg.RANDOM_STATE
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=cfg.N_FEATURES,
        n_informative=cfg.N_INFORMATIVE,
        n_redundant=cfg.N_REDUNDANT,
        weights=weights,
        flip_y=0.01,
        random_state=random_state
    )
    return X, y

def prepare_data_splits(X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test