# Smart-MCQ-Solver-Challenge

Build AI systems that solve &amp; predict top 3 correct answers for challenging multiple choice questions

## Models

| Model Architecture | Parameter Count | Deployment Size | Kaggle MAP@3 Score |
| :--- | :--- | :--- | :--- |
| **Scratch Model** | ~2.6M | **~10 MB** | **0.75561** |
| **Pretrained bert-base-uncased Model** | ~110M | ~440 MB | 0.75103 |
| **Pretrained qwen2.5-0.5b Model** | ~500M | ~1 GB | **0.76226** |

## Dataset

- **Input:** Question prompt + 5 possible answer options (A, B, C, D, E)
- **Objective:** Rank top 3 most likely correct answers

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
transformers
wandb
```

## Training & Inference

The `/src/` directory contains all executable scripts for training and inference. It is the main entry point for running the models.

## Quickstart

1. Install dependencies: `pip install -r requirements.txt`
2. Run scripts from the repository root to ensure model artifacts save to the correct directories. For example, to train the scratch model:
   ```bash
   python /src/scratch_model/train.py
   ```

*Note:* The custom MLP runs easily on CPU. The Transformers (BERT & Qwen) require a CUDA-compatible GPU to avoid memory errors.

