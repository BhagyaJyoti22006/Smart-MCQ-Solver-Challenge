import os
# Prevent transformers from crashing due to broken local TensorFlow dependencies
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizerBase


# 1. Initialization & Setup

# Fix for Qwen's tokenizer_config.json
# Qwen's tokenizer_config.json saves extra_special_tokens as a list instead of a dict causing transformers to crash
original_set_special_tokens = PreTrainedTokenizerBase._set_model_specific_special_tokens
def patched_set_special_tokens(self, special_tokens):
    if isinstance(special_tokens, list):
        # Dynamically convert list to the expected dictionary format in memory
        special_tokens = {token: token for token in special_tokens}
    original_set_special_tokens(self, special_tokens)
PreTrainedTokenizerBase._set_model_specific_special_tokens = patched_set_special_tokens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "pretrained_qwen2.5-0.5b_model"

print(f"Loading tokenizer from {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# Qwen requires an explicit pad token to batch inputs properly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model for Sequence Classification (Outputting 1 score per option)
print(f"Loading model from {MODEL_DIR}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, 
    num_labels=1, 
    trust_remote_code=True
).to(device)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# Set model to evaluation mode (disables dropout layers for consistent inference)
model.eval()


# 2. Prediction Function

# Takes a question and 5 options, scores each (Prompt + Option) pair, and returns their probabilities.
def predict_mcq(prompt, option_A, option_B, option_C, option_D, option_E):
    options = [option_A, option_B, option_C, option_D, option_E]
    
    # Duplicate the prompt 5 times to pair with each option
    first_sentences = [prompt] * 5
    
    # Tokenize the 5 pairs simultaneously (Batch Size = 5)
    tokenized = tokenizer(
        first_sentences,
        options,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Move tokens to GPU/CPU
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Disable gradient calculation for faster inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Reshape the 5 outputs into a single row of 5 logits
        logits = outputs.logits.view(-1, 5)
        # Convert raw logits to percentages (probabilities) using Softmax
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Map probabilities to their respective labels
    labels = ["A", "B", "C", "D", "E"]
    
    return {labels[i]: float(probs[i]) for i in range(5)}


# 3. Gradio Web Interface

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
    description="This is the deployed fine-tuned Qwen2.5-0.5B model from the DLGA Project. It predicts the top 3 most likely answers to any given multiple-choice question.",
    theme="soft",
    flagging_mode="never" # Removes the default Gradio 'Flag' button
)


# Boot up the server when the script is run
if __name__ == "__main__":
    demo.launch()

