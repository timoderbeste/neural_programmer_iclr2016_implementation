import torch
import torch.nn as nn


class HistoryRNN(nn.Module):
    def __init__(self, hidden_size: int,  num_of_operations: int, num_of_columns: int):
        super(HistoryRNN, self).__init__()

        self.hidden_size = hidden_size

        self.operation2hidden = nn.Linear(num_of_operations, hidden_size)
        self.column2hidden = nn.Linear(num_of_columns, hidden_size)
        self.hidden2hidden = nn.Linear(3 * hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, alpha_op: torch.Tensor, alpha_col: torch.Tensor, hidden_state: torch.Tensor):
        return self.tanh(self.hidden2hidden(torch.cat(
            (torch.cat((self.operation2hidden(alpha_op), self.column2hidden(alpha_col))), hidden_state))))
