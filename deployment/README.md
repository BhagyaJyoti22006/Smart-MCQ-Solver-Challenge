# Smart MCQ Solver

This is the deployment repository for the **Deep Learning & Generative AI (DLGA) Project**. 

It contains a fine-tuned version of `Qwen2.5-0.5B` specifically trained to solve complex multiple-choice questions (MCQs). The model takes a question prompt and 5 possible options (A, B, C, D, E) and mathematically predicts the top 3 most likely answers.

## How it Works
1. **Tokenization**: The input question is duplicated 5 times and paired with each of the 5 options.
2. **Inference**: The model processes all 5 pairs simultaneously as a single batch using sequence classification.
3. **Softmax Output**: The raw logits are squashed into percentage probabilities using a Softmax function.
4. **Ranking**: The Gradio interface displays the top 3 highest probabilities to the user.

## Local Testing
To run this exact UI on local machine:
```bash
pip install -r requirements.txt
python app.py
```
*The web app will automatically launch at `http://127.0.0.1:7860`.*

## Repository Structure
- `app.py`: The Gradio web interface and PyTorch inference engine.
- `requirements.txt`: Minimal dependencies needed to run the app.
- `pretrained_qwen2.5-0.5b_model/`: The raw `.safetensors` model weights and tokenizer configurations.

