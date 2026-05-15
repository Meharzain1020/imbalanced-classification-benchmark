import numpy as np
from src.data_loader import generate_imbalanced_data, prepare_data_splits
import src.config as cfg

def test_data_generation_and_ratio():
    X, y = generate_imbalanced_data(n_samples=1000, weights=[0.95, 0.05], random_state=42)
    assert X.shape == (1000, cfg.N_FEATURES)
    assert len(y) == 1000
    
    minority_count = np.sum(y == 1)
    assert 40 <= minority_count <= 60

def test_data_splits():
    X, y = generate_imbalanced_data(n_samples=500, weights=[0.9, 0.1], random_state=42)
    X_train, X_test, y_train, y_test = prepare_data_splits(X, y, test_size=0.2, random_state=42)
    
    assert X_train.shape[0] == 400
    assert X_test.shape[0] == 100
    assert len(y_train) == 400
    assert len(y_test) == 100