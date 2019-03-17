import torch
import torch.nn as nn


class Selector(nn.Module):
    def __init__(self, hidden_dim: int, num_of_operations: int, num_of_columns: int):
        super(Selector, self).__init__()

        self.hidden_dim = hidden_dim

        self.input2operation_hidden = nn.Linear(2 * hidden_dim, hidden_dim)
        self.input2column_hidden = nn.Linear(2 * hidden_dim, hidden_dim)
        self.hidden2operation = nn.Linear(hidden_dim, num_of_operations)
        self.hidden2column = nn.Linear(hidden_dim, num_of_columns)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, encoded_question: torch.Tensor, history: torch.Tensor):
        selector_input = torch.cat((encoded_question, history))
        operation_hidden = self.tanh(self.input2operation_hidden(selector_input))
        column_hidden = self.tanh(self.input2column_hidden(selector_input))
        operation = self.softmax(self.hidden2operation(operation_hidden))
        column = self.softmax(self.hidden2column(column_hidden))

        return operation, column

