import torch
import torch.nn as nn

from src.models.question_rnn import QuestionRNN
from src.models.selector import Selector
from src.models.history_rnn import HistoryRNN
from src.models.operations import calc_scalar_answer, calc_lookup_answer, calc_row_select


class NeuralProgrammer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int,
                 num_of_operations: int, num_of_columns: int, num_time_steps: int):
        super(NeuralProgrammer, self).__init__()
        self.hidden_size = hidden_size
        self.num_time_steps = num_time_steps

        self.question_rnn = QuestionRNN(hidden_size, vocab_size)
        self.selector = Selector(hidden_size, num_of_operations, num_of_columns)
        self.history_rnn = HistoryRNN(hidden_size, num_of_operations, num_of_columns)

        self.U = nn.Embedding(2, hidden_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_question: [int], input_question_numbers: [int], left_word_indices: [int],
                table: torch.Tensor):
        # initializing the values at t = 0
        history_states = [torch.zeros(self.hidden_size)]
        scalar_answers = [torch.tensor(0)]
        lookup_answers = [torch.zeros(table.size(0), table.size(1))]
        row_selects = [torch.ones(table.size(0))]
        z_matrix = torch.zeros(len(input_question_numbers), self.hidden_size)

        input_question_numbers = torch.tensor(input_question_numbers)
        left_word_indices = torch.tensor(left_word_indices)

        # Encoding the input question using the QuestionRNN module.
        encoded_question = self.question_rnn(input_question)
        question_hidden_states = self.question_rnn.hidden_states

        for i in range(len(input_question_numbers)):
            z_matrix[i] = question_hidden_states[left_word_indices[i]]

        for t in range(1, self.num_time_steps + 1):
            # Preparing the useful values
            history_state_past1 = history_states[-1]

            scalar_output_past1 = scalar_answers[-1]
            scalar_output_past3 = scalar_answers[max(0, len(scalar_answers) - 3)]

            row_select_past1 = row_selects[-1]
            row_select_past2 = row_selects[max(0, len(row_selects) - 2)]

            # Selecting the appropriate op and col using the Selector module.
            alpha_op, alpha_col = self.selector(encoded_question, history_state_past1)

            # Calculating row_select
            row_select = torch.zeros(table.size(0))
            beta_lesser = self.softmax(z_matrix.mm(self.U(torch.tensor(0)).view(10, 1)))
            l_pivot = sum([beta_lesser[i] * input_question_numbers[i] for i in range(len(input_question_numbers))])
            beta_greater = self.softmax(z_matrix.mm(self.U(torch.tensor(1)).view(10, 1)))
            g_pivot = sum([beta_greater[i] * input_question_numbers[i] for i in range(len(input_question_numbers))])

            for i in range(table.size(0)):
                row_select[i] = calc_row_select(i, row_select_past1, row_select_past2, table, l_pivot, g_pivot,
                                                alpha_op, alpha_col)
            row_selects.append(row_select)

            # Calculating the scalar answer and lookup answer
            scalar_answers.append(calc_scalar_answer(row_select_past1, scalar_output_past3, scalar_output_past1,
                                                     table, alpha_op))

            lookup_answer = torch.zeros(table.size(0), table.size(1))
            for i in range(table.size(0)):
                for j in range(table.size(1)):
                    lookup_answer[i][j] = calc_lookup_answer(i, j, row_select, alpha_col, alpha_op)
            lookup_answers.append(lookup_answer)

            history_states.append(self.history_rnn(alpha_op, alpha_col, history_state_past1))
            
        return scalar_answers[-1], lookup_answers[-1]


def test():
    input_question = [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]
    input_question_numbers = [1, 2]
    left_word_indices = [0, 3]
    table = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])

    np = NeuralProgrammer(10, 10, 9, 2, 4)
    scalar_answer, lookup_answer = np(input_question, input_question_numbers, left_word_indices, table)

    print(scalar_answer)
    print(lookup_answer)


if __name__ == '__main__':
    test()
