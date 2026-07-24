import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

# Add the root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import load_data, MCQDataset, LABEL_MAP, LABEL_LETTERS, init_wandb, calculate_map3
from src.scratch_model.model import CustomMLP


def extract_features(df, tfidf, is_train=True):
    # Concatenates the prompt with each option and transforms them using TF-IDF.
    # Returns a flattened tensor for each row.
    
    features = []
    print(f"Processing {len(df)} text samples...")
    
    for _, row in df.iterrows():
        row_features = []
        for l in LABEL_LETTERS:
            combined_text = str(row["prompt"]) + " " + str(row[l])
            # If train, just transform as already fitted
            vec = tfidf.transform([combined_text]).toarray()[0]
            row_features.extend(vec)
        features.append(row_features)
    
    return np.array(features, dtype=np.float32)


def main():
    
    # Workflow Reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Configured Compute Device: {device}")
    
    # Loading Datasets
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    train_df, test_df = load_data(data_dir=data_dir)
    
    # Text Vectorization (TF-IDF)
    print("Initializing TF-IDF Vectorizer (max_features=1000)...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    # Build vocabulary on all available text
    all_text = train_df["prompt"].tolist() + test_df["prompt"].tolist()
    for l in LABEL_LETTERS:
        all_text.extend(train_df[l].tolist())
        all_text.extend(test_df[l].tolist())
    tfidf.fit(all_text)
    
    # Save the vectorizer for inference
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "scratch_tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    
    # Feature Extraction & Dataloaders
    X_train = extract_features(train_df, tfidf)
    y_train = np.array([LABEL_MAP[ans] for ans in train_df["answer"]], dtype=np.int64)
    train_dataset = MCQDataset(X_train, y_train)
    input_size = X_train.shape[1]
    
    # Define the distinct HPT Configurations
    hpt_configs = [
        {"run_name": "hpt_baseline", "lr": 1e-3, "epochs": 8, "batch_size": 32, "full_train": False},
        {"run_name": "hpt_lower_lr", "lr": 5e-4, "epochs": 12, "batch_size": 32, "full_train": False},
        {"run_name": "hpt_slow_steady", "lr": 1e-4, "epochs": 15, "batch_size": 32, "full_train": False},
        {"run_name": "hpt_aggressive", "lr": 5e-3, "epochs": 6, "batch_size": 32, "full_train": False},
        {"run_name": "hpt_large_batch", "lr": 1e-3, "epochs": 10, "batch_size": 64, "full_train": False},
        {"run_name": "final_full_train", "lr": 1e-3, "epochs": 8, "batch_size": 32, "full_train": True},
    ]
    
    # Automated Hyperparameter Tuning Loop
    for config in hpt_configs:
        # Format: <model_type>_<parameter_type>_<time in yymmdd_hhmmss>
        timestamp = time.strftime("%y%m%d_%H%M%S")
        unique_run_name = f"scratch_{config['run_name']}_{timestamp}"
        print(f"\n{'='*50}\nStarting Experiment: {unique_run_name}\n{'='*50}")
        
        # Validation Split Logic
        if not config["full_train"]:
            X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
            train_dataset_run = MCQDataset(X_t, y_t)
            val_dataset_run = MCQDataset(X_v, y_v)
            val_loader = DataLoader(val_dataset_run, batch_size=config["batch_size"], shuffle=False)
            print(f"Mode: Validation Split (Train: {len(X_t)}, Val: {len(X_v)})")
        else:
            train_dataset_run = MCQDataset(X_train, y_train)
            val_loader = None
            print(f"Mode: Full Train (Train: {len(X_train)})")
        
        # Recreate the dataloader and model from scratch for each run
        train_loader = DataLoader(train_dataset_run, batch_size=config["batch_size"], shuffle=True)
        model = CustomMLP(input_size=input_size).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        
        # Initialize Weights & Biases
        run = init_wandb(
            run_name=unique_run_name,
            config_dict={
                "architecture": "Custom MLP",
                "epochs": config["epochs"],
                "learning_rate": config["lr"],
                "batch_size": config["batch_size"],
                "full_train": config["full_train"],
                "optimizer": "Adam",
                "features": "TF-IDF (1000 max_features)"
            }
        )
        
        # Inner Training Loop
        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Step 1: Forward pass
                y_pred = model(batch_X)
                # Step 2: Calculate the loss
                batch_loss = loss_fn(y_pred, batch_y)
                # Step 3: Zero gradients from previous iteration
                optimizer.zero_grad()
                # Step 4: Backpropagation (compute new gradients)
                batch_loss.backward()
                # Step 5: Update parameters
                optimizer.step()
                
                # Metric tracking
                running_loss += batch_loss.item() * batch_X.size(0)
                _, predicted = torch.max(y_pred, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            
            # Validation Evaluation Phase
            if not config["full_train"]:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_preds = []
                all_targets = []
                all_probs = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        
                        # Step 1: Forward pass (compute predictions)
                        y_pred = model(batch_X)
                        # Step 2: Calculate the validation loss
                        loss = loss_fn(y_pred, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                        # Step 3: Metric Tracking (gather probabilities and predictions)
                        probs = torch.nn.functional.softmax(y_pred, dim=1)
                        _, predicted = torch.max(probs, 1)
                        
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                        
                        all_preds.extend(predicted.cpu().numpy())
                        all_targets.extend(batch_y.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                
                epoch_val_loss = val_loss / val_total
                epoch_val_acc = val_correct / val_total
                epoch_val_f1 = f1_score(all_targets, all_preds, average='macro')
                epoch_val_map3 = calculate_map3(all_targets, all_probs)
                
                print(f"Epoch [{epoch+1}/{config['epochs']}] | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Val F1: {epoch_val_f1:.4f}, Val MAP@3: {epoch_val_map3:.4f}")
                
                # Log metrics to W&B (Validation Mode)
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": epoch_acc,
                    "val_loss": epoch_val_loss,
                    "val_accuracy": epoch_val_acc,
                    "val_f1": epoch_val_f1,
                    "val_map3": epoch_val_map3
                })
            else:
                print(f"Epoch [{epoch+1}/{config['epochs']}] | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
                
                # Log metrics to W&B (Full Train Mode)
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": epoch_acc
                })
        
        # Save Model Weights individually to be used later
        model_path = os.path.join(models_dir, f"{unique_run_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Close the run so W&B knows this specific experiment is done
        run.finish()


if __name__ == "__main__":
    main()

