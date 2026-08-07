import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import spaces


# 1. Architecture Definition
class CustomMLP(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(CustomMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.network(x)


# 2. Initialization & Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading TF-IDF Vectorizer...")
tfidf = joblib.load("scratch_tfidf_vectorizer.pkl")
# The vectorizer was trained with max_features=1000. 
# Concatenate 5 options per question, so the input size is 1000 * 5 = 5000.
INPUT_SIZE = len(tfidf.vocabulary_) * 5
print(f"Loading Scratch Model (Input Size: {INPUT_SIZE})...")
model = CustomMLP(input_size=INPUT_SIZE).to(device)
model.load_state_dict(torch.load("scratch_model.pth", map_location=device))
model.eval()


# 3. Prediction Engine

@spaces.GPU
def predict_mcq(prompt, option_A, option_B, option_C, option_D, option_E):
    options = [option_A, option_B, option_C, option_D, option_E]
    
    # Feature Extraction (TF-IDF)
    features = []
    for opt in options:
        combined_text = str(prompt) + " " + str(opt)
        # Transform returns a sparse matrix; convert to dense array and extract the 1D vector
        vec = tfidf.transform([combined_text]).toarray()[0]
        features.extend(vec)
    
    # Convert the concatenated features into a PyTorch tensor (Batch Size = 1)
    # Shape becomes [1, 5000]
    input_tensor = torch.tensor([features], dtype=torch.float32).to(device)
    
    # Model Inference
    with torch.no_grad():
        logits = model(input_tensor)
        # Convert raw logits to percentages (probabilities) using Softmax
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Map probabilities to their respective labels
    labels = ["A", "B", "C", "D", "E"]
    
    return {labels[i]: float(probs[i]) for i in range(5)}


# 4. Gradio Web Interface

demo = gr.Interface(
    fn=predict_mcq,
    inputs=[
        gr.Textbox(lines=3, label="Question Prompt", placeholder="Enter the question here..."),
        gr.Textbox(lines=1, label="Option A"),
        gr.Textbox(lines=1, label="Option B"),
        gr.Textbox(lines=1, label="Option C"),
        gr.Textbox(lines=1, label="Option D"),
        gr.Textbox(lines=1, label="Option E"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Top 3 Predictions"),
    title="Smart MCQ Solver",
    description="This is the deployed PyTorch Scratch Model from the DLGA Project. It uses a custom TF-IDF Vectorizer and a Multi-Layer Perceptron (MLP) to predict the top 3 most likely answers to any multiple-choice question.",
    flagging_mode="never" # Removes the default Gradio 'Flag' button
)


# Boot up the server when the script is run
if __name__ == "__main__":
    demo.launch()

