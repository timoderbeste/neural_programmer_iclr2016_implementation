import torch
import torch.nn as nn

from src.models.question_rnn import QuestionRNN
from src.models.selector import Selector
from src.models.history_rnn import HistoryRNN
from src.models.operations import calc_scalar_answer, calc_lookup_answer, calc_row_select


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class NeuralProgrammer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int,
                 num_of_operations: int, num_of_columns: int, num_time_steps: int):
        super(NeuralProgrammer, self).__init__()
        self.hidden_size = hidden_size
        self.num_time_steps = num_time_steps

        self.question_rnn = QuestionRNN(hidden_size, vocab_size)
        self.question_rnn.to(device)
        self.selector = Selector(hidden_size, num_of_operations, num_of_columns)
        self.selector.to(device)
        self.history_rnn = HistoryRNN(hidden_size, num_of_operations, num_of_columns)
        self.history_rnn.to(device)

        self.U = nn.Embedding(2, hidden_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_question: [int], input_question_numbers: [int], left_word_indices: [int],
                table: torch.Tensor, mode: str = 'train'):
        # initializing the values at t = 0
        history_states = [torch.zeros(self.hidden_size).to(device)]
        scalar_answers = [torch.tensor(0.).to(device)]
        lookup_answers = [torch.zeros(table.size(0), table.size(1)).to(device)]
        row_selects = [torch.ones(table.size(0)).to(device)]
        z_matrix = torch.zeros(len(input_question_numbers), self.hidden_size).to(device)

        input_question_numbers = torch.tensor(input_question_numbers).to(device)
        left_word_indices = torch.tensor(left_word_indices).to(device)

        # Encoding the input question using the QuestionRNN module.
        encoded_question = self.question_rnn(input_question)
        question_hidden_states = self.question_rnn.hidden_states

        for i in range(len(input_question_numbers)):
            z_matrix[i] = question_hidden_states[int(left_word_indices[i])]

        for t in range(1, self.num_time_steps + 1):
            # Preparing the useful values
            history_state_past1 = history_states[-1]

            scalar_output_past1 = scalar_answers[-1]
            scalar_output_past3 = scalar_answers[max(0, len(scalar_answers) - 3)]

            row_select_past1 = row_selects[-1]
            row_select_past2 = row_selects[max(0, len(row_selects) - 2)]

            # Selecting the appropriate op and col using the Selector module.
            alpha_op, alpha_col = self.selector(encoded_question, history_state_past1, mode)

            # Calculating row_select
            row_select = torch.zeros(table.size(0)).to(device)
            if len(input_question_numbers) != 0:
                if mode == 'train':
                    beta_lesser = self.softmax(z_matrix.mm(self.U(torch.tensor(0).to(device)).view(256, 1)))
                else:
                    beta_lesser = torch.zeros(z_matrix.mm(self.U(torch.tensor(0).to(device)).view(256, 1)).size())
                    beta_lesser[int(torch.argmax(z_matrix.mm(self.U(torch.tensor(0).to(device)).view(256, 1))))] = 1.
                l_pivot = sum([beta_lesser[i] * input_question_numbers[i] for i in range(len(input_question_numbers))])
                if mode == 'train':
                    beta_greater = self.softmax(z_matrix.mm(self.U(torch.tensor(1).to(device)).view(256, 1)))
                else:
                    beta_greater = torch.zeros(z_matrix.mm(self.U(torch.tensor(1).to(device)).view(256, 1)).size())
                    beta_greater[int(torch.argmax(z_matrix.mm(self.U(torch.tensor(1).to(device)).view(256, 1))))] = 1.
                g_pivot = sum([beta_greater[i] * input_question_numbers[i] for i in range(len(input_question_numbers))])
            else:
                l_pivot = torch.tensor(-1.).to(device)
                g_pivot = torch.tensor(-1.).to(device)

            for i in range(table.size(0)):
                row_select[i] = calc_row_select(i, row_select_past1, row_select_past2, table, l_pivot, g_pivot,
                                                alpha_op, alpha_col)
            row_selects.append(row_select)

            # Calculating the scalar answer and lookup answer
            scalar_answer = calc_scalar_answer(row_select_past1, scalar_output_past3, scalar_output_past1,
                                               table, alpha_op, alpha_col)
            scalar_answers.append(scalar_answer)

            # TODO needs to check if it is okay to do this... If this will break a graph
            lookup_answer = torch.zeros(table.size(0), table.size(1)).to(device)
            for i in range(table.size(0)):
                for j in range(table.size(1)):
                    lookup_answer[i][j] = calc_lookup_answer(i, j, row_select, alpha_col, alpha_op)
            lookup_answers.append(lookup_answer)

            history_state = self.history_rnn(alpha_op, alpha_col, history_state_past1)
            history_states.append(history_state)

        if mode == 'train':
            return scalar_answers[-1], lookup_answers[-1]
        elif mode == 'eval':
            if torch.all(torch.eq(lookup_answers[-1], lookup_answers[-2])):
                return scalar_answers[-1], True
            else:
                return lookup_answers[-1], False


def test_backward():
    input_question = [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]
    input_question_numbers = [1.]
    left_word_indices = [2.]
    table = [[1.], [3.], [5.]]
    table = torch.tensor(table)

    np = NeuralProgrammer(256, 10, 9, 1, 4)
    scalar_answer, lookup_answer = np(input_question, input_question_numbers, left_word_indices, table)
    print(scalar_answer)
    scalar_answer.backward()
    print(list(np.parameters())[0].grad)


if __name__ == '__main__':
    test_backward()
