import torch
from torch.autograd import Variable

from src.models.neural_programmer import NeuralProgrammer
from src.models.operations import OPERATIONS
from src.data_generators.generate_datasets import generate_single_column_table_dataset
from src.utils import is_number


def preprocess_data(data):
    preprocessed_questions = []
    all_question_numbers = []
    all_left_word_indices = []

    vocab = dict()

    for _, question, _ in data:
        question_numbers = []
        left_word_indices = []
        question = question.split()
        for i in range(len(question)):
            if is_number(question[i]):
                # Assume that number will never occurs at the first place.
                question_numbers.append(int(question[i]))
                question[i] = 'NUM'
                left_word_indices.append(i - 1)

        all_question_numbers.append(question_numbers)
        all_left_word_indices.append(left_word_indices)

        preprocessed_question = []
        for token in question:
            if token not in vocab:
                vocab[token] = len(vocab)
            preprocessed_question.append(vocab[token])
        preprocessed_questions.append(preprocessed_question)

    return vocab, preprocessed_questions, all_question_numbers, all_left_word_indices


def scalar_loss(guess, answer, huber):
    guess = Variable(guess, requires_grad=True)
    answer = Variable(answer)
    huber = Variable(torch.tensor(huber))

    a = abs(guess - answer)
    if a <= huber:
        return 0.5 * pow(a, 2)
    else:
        return huber * a - 0.5 * pow(huber, 2)


def lookup_loss(guess: torch.Tensor, answer: torch.Tensor):
    assert guess.size() == answer.size()
    guess = Variable(guess, requires_grad=True)
    answer = Variable(answer)

    print([answer[i] * torch.log(guess[i]) +
           (torch.log(Variable(torch.tensor(1.))) - answer[i]) * torch.log(Variable(torch.tensor(1.))) - guess[i]
           for i in range(answer.size(0))])

    if answer.dim() == 1:
        return (-1 / torch.tensor(answer.size(0), dtype=torch.float)) * sum(
            [answer[i] * torch.log(guess[i]) +
             (torch.log(Variable(torch.tensor(1.))) - answer[i]) * torch.log(Variable(torch.tensor(1.))) - guess[i]
             for i in range(answer.size(0))]
        )


def loss_fn(scalar_guess, lookup_guess, answer, is_scalar):
    return scalar_loss(scalar_guess, answer, 10.) if is_scalar else lookup_loss(lookup_guess, answer)


def main():

    # TODO implement a mini-batch technique for training
    tables, data = generate_single_column_table_dataset(100000, 1, 20)
    vocab, preprocessed_questions, all_question_numbers, all_left_word_indices = \
        preprocess_data(data)

    model = NeuralProgrammer(256, len(vocab), len(OPERATIONS), 1, 4)
    optimizer = torch.optim.Adam(model.parameters())

    for i in range(len(preprocessed_questions)):
        preprocessed_question = torch.tensor(preprocessed_questions[i])
        question_numbers = torch.tensor(all_question_numbers[i])
        left_word_indices = torch.tensor(all_left_word_indices[i])
        answer, is_scalar = torch.tensor(data[i][2][0]), torch.tensor(data[i][2][1])
        table = torch.tensor(tables[data[i][0]]).t()

        model.zero_grad()
        scalar_guess, lookup_guess = model.train_step(preprocessed_question, question_numbers, left_word_indices, table)
        loss = loss_fn(scalar_guess, lookup_guess, answer, is_scalar)
        print(loss)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
