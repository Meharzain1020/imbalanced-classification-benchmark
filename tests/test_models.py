import torch
from src.models import ImbalancedMLP, FocalLoss

def test_mlp_forward_pass():
    model = ImbalancedMLP(input_dim=20)
    x = torch.randn(16, 20)
    out = model(x)
    assert out.shape == (16, 1)

def test_focal_loss_output():
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(10, 1)
    targets = torch.randint(0, 2, (10, 1)).float()
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0.0