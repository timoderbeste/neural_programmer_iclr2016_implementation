def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def build_vocab(question_dicts):
    vocab = dict()
    vocab['NUM'] = 0

    for question_dict in question_dicts:
        question = question_dict['question']
        question = question.split()
        for token in question:
            if token not in vocab and not is_number(token):
                vocab[token] = len(vocab)

    return vocab


def preprocess_data(vocab, question_dicts):
    preprocessed_questions = []
    all_question_numbers = []
    all_left_word_indices = []

    for question_dict in question_dicts:
        question_numbers = []
        left_word_indices = []
        question = question_dict['question']
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
            preprocessed_question.append(vocab[token])
        preprocessed_questions.append(preprocessed_question)

    return preprocessed_questions, all_question_numbers, all_left_word_indices