import os
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add the root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils import load_data, init_wandb, calculate_map3
from src.pretrained_model.dataset import HFMCQDataset
from src.pretrained_model.model import get_pretrained_model


# Define the pretrained architecture to be used across all HPT runs.
PRETRAINED_MODEL = "bert-base-uncased"

def main():
    
    # Workflow Reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Configured Compute Device: {device}")
    print(f"Target Architecture: {PRETRAINED_MODEL}")
    
    # Loading Datasets
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    train_df, test_df = load_data(data_dir=data_dir)
    
    # Define the distinct HPT Configurations
    hpt_configs = [
        # Validation Split Runs (80/20) for Metric Tracking
        {"run_name": f"val_baseline", "lr": 2e-5, "epochs": 3, "batch_size": 4, "full_train": False},
        {"run_name": f"val_lower_lr", "lr": 1e-5, "epochs": 4, "batch_size": 4, "full_train": False},
        {"run_name": f"val_higher_lr", "lr": 3e-5, "epochs": 2, "batch_size": 4, "full_train": False},
        {"run_name": f"val_larger_batch", "lr": 2e-5, "epochs": 3, "batch_size": 8, "full_train": False},
        {"run_name": f"val_more_epochs", "lr": 2e-5, "epochs": 5, "batch_size": 4, "full_train": False},
        
        # Full 100% Training Runs (Kaggle Submissions)
        {"run_name": f"full_baseline", "lr": 2e-5, "epochs": 3, "batch_size": 4, "full_train": True},
        {"run_name": f"full_lower_lr", "lr": 1e-5, "epochs": 4, "batch_size": 4, "full_train": True},
        {"run_name": f"full_higher_lr", "lr": 3e-5, "epochs": 2, "batch_size": 4, "full_train": True},
        {"run_name": f"full_larger_batch", "lr": 2e-5, "epochs": 3, "batch_size": 8, "full_train": True},
        {"run_name": f"full_more_epochs", "lr": 2e-5, "epochs": 5, "batch_size": 4, "full_train": True},
    ]
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # Automated Hyperparameter Tuning Loop
    for config in hpt_configs:
        # Format: <model_type>_<parameter_type>_<time in yymmdd_hhmmss>
        timestamp = time.strftime("%y%m%d_%H%M%S")
        unique_run_name = f"pretrained_{PRETRAINED_MODEL}_{config['run_name']}_{timestamp}"
        print(f"\n{'='*50}\nStarting Experiment: {unique_run_name}\n{'='*50}")
        
        # Load Hugging Face Tokenizer (Must happen before dataset creation)
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)
        
        # Validation Split Logic
        if not config["full_train"]:
            train_sub, val_sub = train_test_split(train_df, test_size=0.2, random_state=SEED)
            train_dataset_run = HFMCQDataset(train_sub, tokenizer)
            val_dataset_run = HFMCQDataset(val_sub, tokenizer)
            val_loader = DataLoader(val_dataset_run, batch_size=config["batch_size"], shuffle=False)
            print(f"Mode: Validation Split (Train: {len(train_sub)}, Val: {len(val_sub)})")
        else:
            train_dataset_run = HFMCQDataset(train_df, tokenizer)
            val_loader = None
            print(f"Mode: Full Train (Train: {len(train_df)})")
        
        # Recreate the dataloader and model from scratch for each run
        train_loader = DataLoader(train_dataset_run, batch_size=config["batch_size"], shuffle=True)
        model = get_pretrained_model(PRETRAINED_MODEL).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
        
        # Initialize Weights & Biases
        run = init_wandb(
            run_name=unique_run_name,
            config_dict={
                "architecture": PRETRAINED_MODEL,
                "epochs": config["epochs"],
                "learning_rate": config["lr"],
                "batch_size": config["batch_size"],
                "full_train": config["full_train"],
                "optimizer": "AdamW",
                "features": "Hugging Face Tokenizer"
            }
        )
        
        # Inner Training Loop
        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Step 1: Forward pass
                outputs = model(**batch)
                logits = outputs.logits
                # Step 2: Calculate the loss
                batch_loss = outputs.loss
                # Step 3: Zero gradients from previous iteration
                optimizer.zero_grad()
                # Step 4: Backpropagation (compute new gradients)
                batch_loss.backward()
                # Step 5: Update parameters
                optimizer.step()
                
                # Metric tracking
                running_loss += batch_loss.item() * batch["input_ids"].size(0)
                _, predicted = torch.max(logits, 1)
                total += batch["labels"].size(0)
                correct += (predicted == batch["labels"]).sum().item()
            
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
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        
                        # Step 1: Forward pass (compute predictions)
                        outputs = model(**batch)
                        logits = outputs.logits
                        # Step 2: Calculate the validation loss
                        loss = outputs.loss
                        val_loss += loss.item() * batch["input_ids"].size(0)
                        # Step 3: Metric Tracking (gather probabilities and predictions)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        _, predicted = torch.max(probs, 1)
                        
                        val_total += batch["labels"].size(0)
                        val_correct += (predicted == batch["labels"]).sum().item()
                        
                        all_preds.extend(predicted.cpu().numpy())
                        all_targets.extend(batch["labels"].cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())
                
                epoch_val_loss = val_loss / val_total
                epoch_val_acc = val_correct / val_total
                epoch_val_f1 = f1_score(all_targets, all_preds, average="macro")
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
        
        # Save model wrapper (using hugging face's native save_pretrained)
        model_path = os.path.join(models_dir, unique_run_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model saved to {model_path}")
        
        # Close the run so W&B knows this specific experiment is done
        run.finish()


if __name__ == "__main__":
    main()

