import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import src.config as cfg
from src.models import ImbalancedMLP, FocalLoss

def train_random_forest(X_train, y_train, strategy="baseline"):
    if strategy == "smote":
        smote = SMOTE(random_state=cfg.RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        clf = RandomForestClassifier(n_estimators=100, random_state=cfg.RANDOM_STATE)
    elif strategy == "class_weight":
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=cfg.RANDOM_STATE)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=cfg.RANDOM_STATE)
        
    clf.fit(X_train, y_train)
    return clf

def train_pytorch_mlp(X_train, y_train, strategy="baseline", epochs=cfg.EPOCHS, batch_size=cfg.BATCH_SIZE, lr=cfg.LEARNING_RATE):
    if strategy == "smote":
        smote = SMOTE(random_state=cfg.RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ImbalancedMLP(input_dim=X_train.shape[1]).to(cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    if strategy == "focal_loss":
        criterion = FocalLoss(alpha=cfg.FOCAL_ALPHA, gamma=cfg.FOCAL_GAMMA)
    elif strategy == "class_weight":
        pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)], dtype=torch.float32).to(cfg.DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(cfg.DEVICE), by.to(cfg.DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    return model