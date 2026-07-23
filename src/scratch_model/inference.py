import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pickle

# Add the root directory to the python path so as to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import load_data, LABEL_LETTERS
from src.scratch_model.model import CustomMLP
from src.scratch_model.train import extract_features


def main():
    # Device Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Configured Compute Device: {device}")
    
    # Load Test Data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    _, test_df = load_data(data_dir=data_dir)
    
    # Load Saved Vectorizer
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    vectorizer_path = os.path.join(models_dir, "scratch_tfidf_vectorizer.pkl")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Missing vectorizer at {vectorizer_path}. Did you run train.py first?")
    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)
    print("Loaded saved TF-IDF vectorizer.")
    
    # Feature Extraction
    X_test = extract_features(test_df, tfidf, is_train=False)
    
    # Load Trained Model
    model_path = os.path.join(models_dir, "scratch_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model weights at {model_path}. Did you run train.py first?")
    
    # Must instantiate the model with the exact same input_size
    input_size = X_test.shape[1]
    model = CustomMLP(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded model weights successfully.")
    
    # Inference Loop
    sub_ids = []
    sub_preds = []
    print("\nExecuting Inference on Test Set...")
    
    with torch.no_grad():
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            idx_val = row["id"] if "id" in row else i
            inputs = torch.tensor(X_test[i]).unsqueeze(0).to(device)
            # Forward pass and probability distribution
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            # Extract Top-3 predicted options
            top3_idx = torch.topk(probs, 3, dim=-1).indices[0]
            top3_letters = [LABEL_LETTERS[idx] for idx in top3_idx]
            sub_ids.append(idx_val)
            sub_preds.append(" ".join(top3_letters))
    
    # Generate Submission
    submission = pd.DataFrame({"id": sub_ids, "prediction": sub_preds})
    
    # Save submission in the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    submission_path = os.path.join(project_root, "submission.csv")
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission file generated successfully. Records: {len(submission)}")
    print(f"Saved to: {submission_path}")


if __name__ == "__main__":
    main()

