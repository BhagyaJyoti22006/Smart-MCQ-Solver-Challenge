from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification
)


def get_pretrained_model(model_name, model_type="auto"):
    if model_type == "auto":
        # Qwen and other decoder-only LLMs don't have a MultipleChoice head.
        if "qwen" in model_name.lower():
            model_type = "sequence_classification"
        else:
            model_type = "multiple_choice"
    
    print(f"Loading Hugging Face model: {model_name}")
    
    if model_type == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            trust_remote_code=True,
        )
    elif model_type == "multiple_choice":
        model = AutoModelForMultipleChoice.from_pretrained(
            model_name
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from {'auto', 'multiple_choice', 'sequence_classification'}."
        )
    
    print("Model loaded successfully!")
    return model

