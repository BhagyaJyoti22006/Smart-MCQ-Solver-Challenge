---
title: Smart MCQ Solver
emoji: 🧠
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Smart MCQ Solver

This is the deployment repository for the **Deep Learning & Generative AI (DLGA) Project**. 

It contains a completely custom **Multi-Layer Perceptron (MLP)** model built and trained entirely from scratch in PyTorch to solve multiple-choice questions (MCQs). The model takes a question prompt and 5 possible options (A, B, C, D, E), and mathematically predicts the top 3 most likely answers.

## 🚀 Architecture
This model does not rely on massive pre-trained embeddings like BERT or Qwen. Instead, it uses a lightweight, custom architecture:
1. **Feature Extraction (TF-IDF)**: The input question and options are concatenated and transformed into statistical vectors using a custom-trained TF-IDF vectorizer (1000 vocab).
2. **Concatenation**: The 5 vectors are flattened into a single 5000-dimensional input vector.
3. **MLP Inference**: The vector is passed through a 3-layer Neural Network (Linear -> ReLU -> Dropout) to output 5 raw logits.
4. **Softmax Ranking**: The logits are squashed into percentage probabilities using a Softmax function, and the top 3 highest probabilities are displayed to the user.

> Note: If an input question contains zero words from the vectorizer's 1,000-word science vocabulary, the input vector becomes all zeros, resulting in the model outputting its baseline dataset bias: E (37%), B (21%), A (15%).

## 🛠️ Local Testing
To run this exact UI on local machine:
```bash
pip install -r requirements.txt
python app.py
```
*The web app will automatically launch at `http://127.0.0.1:7860`.*

## 📂 Repository Structure
- `app.py`: The Gradio web interface and PyTorch inference engine.
- `requirements.txt`: Minimal dependencies needed to run the app (`joblib`, `scikit-learn`, `torch`, `gradio`).
- `scratch_model.pth`: The trained PyTorch model weights.
- `scratch_tfidf_vectorizer.pkl`: The statistical vocabulary vectorizer mapping.

