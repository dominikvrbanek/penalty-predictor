#model.py
import torch
import torch.nn as nn

# Inside your training script
weights = torch.tensor([0.823, 3.115, 0.683])  # L, C, R order
criterion = nn.CrossEntropyLoss(weight=weights.to(device))


# You'll use these later in your loss function
# Inverse frequency weighting
total = 832
weights = {
    'L': total / (3 * 337),   # 0.823
    'C': total / (3 * 89),    # 3.115  <-- center gets heavily upweighted
    'R': total / (3 * 406),   # 0.683
}