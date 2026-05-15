import torch


N_SAMPLES = 10000
N_FEATURES = 20
N_INFORMATIVE = 12
N_REDUNDANT = 4
IMBALANCE_RATIO = [0.97, 0.03]
RANDOM_STATE = 42
TEST_SIZE = 0.2


N_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


OUTPUT_DIR = "outputs"
FIGURES_DIR = "outputs/figures"
RESULTS_CSV = "outputs/benchmark_results.csv"