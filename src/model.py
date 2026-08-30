#model.py
import torch
import torch.nn as nn
'''
# Inside your training script
weights = torch.tensor([0.823, 3.115, 0.683])  # L, C, R order
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# You'll use these later in your loss function
# Inverse frequency weighting

Total clips after exclusion: 832
Direction distribution: Counter({'R': 406, 'L': 337, 'C': 89})
Foot distribution: Counter({'right': 584, 'left': 248})

total = 832
weights = {
    'L': total / (3 * 337),   # 0.823
    'C': total / (3 * 89),    # 3.115  <-- center gets heavily upweighted
    'R': total / (3 * 406),   # 0.683
}
'''

import torch
import torch.nn as nn

class PenaltyLSTM(nn.Module):
    def __init__(self, input_size=30, hidden_size=128, num_layers=2, num_classes=3, dropout=0.3):
        super(PenaltyLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # input shape: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :] # (batch, hidden_size)

        logits = self.classifer(last_hidden) # (batch, 3)
        return logits
    