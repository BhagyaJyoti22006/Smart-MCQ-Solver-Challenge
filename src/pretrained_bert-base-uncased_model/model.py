from transformers import AutoModelForMultipleChoice

def get_pretrained_model(model_name="bert-base-uncased"):
    """
    Downloads and initializes the pretrained transformer with a Multiple Choice classification head.
    The head outputs exactly 1 logit per sequence pair.
    Since passing 5 pairs per question, it will output 5 logits.
    """
    print(f"Loading Hugging Face model: {model_name}")
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    print("Model loaded successfully!")
    return model

