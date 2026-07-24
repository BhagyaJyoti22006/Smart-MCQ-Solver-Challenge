import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Shared label mappings
LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
LABEL_LETTERS = ["A", "B", "C", "D", "E"]


def load_data(data_dir="../data"):
    """
    Loads train and test datasets.
    Tries multiple paths to accommodate local and Kaggle and Colab environments.
    """
    possible_paths = [
        "/kaggle/input/competitions/smart-mcq-solver-challenge/",  # Kaggle
        "/content/",                                               # Google Colab
        data_dir,
        "../" # Fallback
    ]
    
    train_df, test_df = None, None
    for path in possible_paths:
        train_path = os.path.join(path, "train.csv")
        test_path = os.path.join(path, "test.csv")
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_df = pd.read_csv(train_path).fillna("")
            test_df = pd.read_csv(test_path).fillna("")
            print(f"Loaded data from {path}")
            break
    
    if train_df is None or test_df is None:
        raise FileNotFoundError("Could not find train.csv and test.csv in the expected directories.")
    
    return train_df, test_df


class MCQDataset(Dataset):
    """
    Standard PyTorch Dataset for Multiple Choice Questions.
    Expects numerical tensors (X) and optional labels (y).
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.int64)
        else:
            self.y = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def calculate_map3(true_labels, predicted_probs):
    """
    Calculates Mean Average Precision @ 3 (MAP@3).
    true_labels: list or array of true class indices
    predicted_probs: 2D array of probability distributions across classes
    """
    map3_score = 0.0
    for true_label, probs in zip(true_labels, predicted_probs):
        # Get indices of top 3 predictions (sorted highest to lowest probability)
        top3_preds = np.argsort(probs)[-3:][::-1]
        # Calculate AP@3 for this sample
        for i, p in enumerate(top3_preds):
            if p == true_label:
                map3_score += 1.0 / (i + 1)
                break
    
    return map3_score / len(true_labels)


def init_wandb(run_name, config_dict=None):
    """
    Initializes a Weights & Biases run.
    """
    import wandb
    
    # Check if W&B API key is set, otherwise don't crash, just warn
    if "WANDB_API_KEY" not in os.environ:
        print("WARNING: WANDB_API_KEY environment variable not set. wandb might prompt for login.")
    
    run = wandb.init(
        entity="24f2002505-dl-genai-project",
        project="24f2002505-t22026",
        name=run_name,
        config=config_dict
    )
    print(f"W&B tracking initialized for {run_name}!")
    return run

