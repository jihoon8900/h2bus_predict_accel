import torch
import torch.nn as nn
import torch.optim as optim


class simpleLSTM(nn.Module):
  def __init__(self, input_dim=1, hidden_dim=32, target_dim=1):
    super().__init__()
    self.lstm = nn.LSTM(input_size=input_dim,
                        hidden_size=hidden_dim, batch_first=True)
    self.hidden2pred = nn.Linear(hidden_dim, target_dim)

  def forward(self, x):
    lstm_out, _ = self.lstm(x)
    net_out = self.hidden2pred(lstm_out)

    return net_out
