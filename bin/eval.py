import json
import torch
from src.utils import preprocess_data


device = torch.device('cpu')


def main():
    print('Loading model and dataset...')
    with open('../data/vocab.txt', 'r') as f:
        vocab = json.load(f)
    model = torch.load('../models/trained_model.pt')
    file_name = '../data/testing_set.txt'
    with open(file_name, 'r') as f:
        question_dicts = json.load(f)

    preprocessed_questions, all_question_numbers, all_left_word_indices = preprocess_data(vocab, question_dicts)

    for i in range(len(preprocessed_questions)):
        preprocessed_question = preprocessed_questions[i]
        question_numbers = all_question_numbers[i]
        left_word_indices = all_left_word_indices[i]
        answer = torch.tensor(question_dicts[i]['answer']).to(device)
        is_scalar = question_dicts[i]['answer_type']
        table = torch.tensor(question_dicts[i]['table']).t().to(device)

        guess = model(preprocessed_question, question_numbers, left_word_indices, table, mode='eval')
        print('The question is:', question_dicts[i]['question'])
        if (is_scalar and not guess[1]) or (not is_scalar and guess[1]) or (not torch.eq(answer.float(), guess[0])):
            print('False, the correct answer is', answer, 'but the guess is', guess)
        else:
            print('Correct!')
        print('=' * 20)


if __name__ == '__main__':
    main()
