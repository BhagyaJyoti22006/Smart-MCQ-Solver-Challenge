import torch.nn as nn

class CustomMLP(nn.Module):
    """
    Multi-Layer Perceptron trained from scratch for multiple-choice QA.
    Uses extensive Dropout to prevent overfitting on small datasets.
    """
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

