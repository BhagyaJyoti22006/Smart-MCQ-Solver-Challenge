# Best model name
BEST_MODEL_DIRNAME  = "pretrained_qwen2.5-0.5b_full_larger_batch_260729_084136"


import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add the root directory to the python path so as to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import load_data, LABEL_LETTERS
from src.pretrained_model.dataset import HFMCQDataset


def main():
    # Device Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Configured Compute Device: {device}")
    
    # Load Test Data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    _, test_df = load_data(data_dir=data_dir)
    
    # Load Trained Model
    models_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    model_dirname = BEST_MODEL_DIRNAME
    if model_dirname == "LATEST":
        # Find the most recently modified directory that starts with 'pretrained_'
        pretrained_dirs = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d)) and d.startswith("pretrained_")]
        if not pretrained_dirs:
            raise FileNotFoundError("No pretrained model directories found in models/")
        model_dirname = max(pretrained_dirs, key=lambda d: os.path.getmtime(os.path.join(models_root, d)))
        print(f"Auto-selected latest model: {model_dirname}")
    models_dir = os.path.join(models_root, model_dirname)
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Missing model directory at {models_dir}. Update BEST_MODEL_DIRNAME at the top of the script!")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(models_dir)
    print("Loaded saved Hugging Face Tokenizer.")
    
    # Feature Extraction (via Dataset)
    test_dataset = HFMCQDataset(test_df, tokenizer, is_test=True)
    
    # Load Trained Model
    model = AutoModelForSequenceClassification.from_pretrained(models_dir).to(device)
    model.eval()
    print("Loaded model weights successfully.")
    
    # Inference Loop
    sub_ids = []
    sub_preds = []
    print("\nExecuting Inference on Test Set...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            row = test_df.iloc[i]
            idx_val = row["id"] if "id" in row else i
            
            # Get the tensor dictionary for this specific row and add batch dimension
            item = test_dataset[i]
            inputs = {k: v.unsqueeze(0).to(device) for k, v in item.items()}
            
            # Forward pass and probability distribution
            input_ids = inputs["input_ids"].view(-1, inputs["input_ids"].size(-1))
            attention_mask = inputs["attention_mask"].view(-1, inputs["attention_mask"].size(-1))
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1, 5)
            probs = F.softmax(logits, dim=1)
            
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

