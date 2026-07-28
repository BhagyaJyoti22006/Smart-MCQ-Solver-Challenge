import torch
from torch.utils.data import Dataset

class HFMCQDataset(Dataset):
    """
    Dataset class designed specifically for Hugging Face AutoModelForMultipleChoice.
    It takes the prompt and pairs it with each of the 5 options, creating 5 sequence pairs per question.
    """
    def __init__(self, df, tokenizer, max_length=128, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        self.options = ["A", "B", "C", "D", "E"]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row["prompt"])
        
        # Create 5 pairs of (prompt, option_text)
        first_sentences = [prompt] * 5
        second_sentences = [str(row[opt]) for opt in self.options]
        
        # Tokenize all 5 pairs for this specific question
        tokenized = self.tokenizer(
            first_sentences,
            second_sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Hugging Face models expect a dictionary of tensors (input_ids, attention_mask)
        item = {key: val for key, val in tokenized.items()}
        
        # If this is not the hidden test set, include the target label
        if not self.is_test:
            label = self.label_map[row["answer"]]
            item["labels"] = torch.tensor(label, dtype=torch.long)
        
        return item

