import json

import tqdm
import torch

from src.models.neural_programmer import NeuralProgrammer
from src.models.operations import OPERATIONS
from src.utils import build_vocab, preprocess_data

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def scalar_loss(guess, answer, huber):
    huber = torch.tensor(huber).to(device)

    a = abs(guess - answer)
    if a <= huber:
        return 0.5 * pow(a, 2)
    else:
        return huber * a - 0.5 * pow(huber, 2)


def lookup_loss(guess: torch.Tensor, answer: torch.Tensor):
    assert guess.size() == answer.size()

    return (-1 / (torch.tensor(answer.size(0), dtype=torch.float)) * (
        torch.tensor(answer.size(1), dtype=torch.float))) * sum(
        [
            answer[i][j] * torch.log(guess[i][j] + torch.tensor(0.00000001)) +
            (torch.tensor(1.) - answer[i][j]) * torch.log(torch.tensor(1.) - guess[i][j])
            for j in range(answer.size(1)) for i in range(answer.size(0))
        ]
    )


def loss_fn(scalar_guess, lookup_guess, answer, is_scalar):
    return scalar_loss(scalar_guess, answer, 10.) if is_scalar else lookup_loss(lookup_guess, answer)


def main():
    # TODO implement a mini-batch technique for training
    print('Loading dataset...')
    file_name = '../data/training_set.txt'
    with open(file_name, 'r') as f:
        question_dicts = json.load(f)

    print('Pre-processing the questions...')
    vocab = build_vocab(question_dicts)
    with open('../data/vocab.txt', 'w') as f:
        json.dump(vocab, f)

    preprocessed_questions, all_question_numbers, all_left_word_indices = \
        preprocess_data(vocab, question_dicts)

    model = NeuralProgrammer(256, len(vocab), len(OPERATIONS), 1, 4)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print('Starting to train...')
    for epoch in range(10):
        total_loss = 0.0
        for i in tqdm.tqdm(range(len(preprocessed_questions))):
            preprocessed_question = preprocessed_questions[i]
            question_numbers = all_question_numbers[i]
            left_word_indices = all_left_word_indices[i]
            answer = torch.tensor(question_dicts[i]['answer']).to(device)
            # is_scalar = torch.tensor(question_dicts[i]['answer_type'])
            table = torch.tensor(question_dicts[i]['table']).t().to(device)

            scalar_guess, lookup_guess = model(preprocessed_question, question_numbers, left_word_indices, table,
                                               mode='train')
            loss = scalar_loss(scalar_guess, answer, 10.)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(('avg loss at epoch %d: ' % epoch), total_loss / len(preprocessed_question))

        if epoch % 5 == 0 and epoch != 0:
            print('Saving model at epoch %d' % epoch)
            torch.save(model, '../models/trained_model_epoch%d.pt' % epoch)

    torch.save(model, '../models/trained_model.pt')


if __name__ == '__main__':
    main()
