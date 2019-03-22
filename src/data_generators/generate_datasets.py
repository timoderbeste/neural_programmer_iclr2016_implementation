import random

import src.data_generators.generate_questions as gq
import src.data_generators.generate_tables as gt


def generate_single_column_table_dataset(size: int, min_limit: int, max_limit: int):
    generated_tables = []
    generated_data = []

    while len(generated_data) < size:
        table_idx = random.randint(0, len(generated_tables))
        if table_idx < len(generated_tables):
            table = generated_tables[table_idx]
        else:
            table = gt.generate_single_column_table(min_limit, max_limit)
            generated_tables.append(table)

        question_generation_function = \
            gq.QUESTION_GENERATION_FUNCTIONS[random.randint(0, len(gq.QUESTION_GENERATION_FUNCTIONS) - 1)]

        if question_generation_function == gq.generate_single_column_table_sum_all:
            question, answer = question_generation_function(table)
        elif question_generation_function == gq.generate_single_column_table_count_all:
            question, answer = question_generation_function(table)
        # elif question_generation_function == gq.generate_single_column_table_sum_greater:
        else:
            question, answer = question_generation_function(table, min_limit, max_limit)

        generated_data.append((table_idx, question, answer))

    return generated_tables, generated_data


def test():
    tables, data = generate_single_column_table_dataset(10, 0, 10)
    print(tables)
    print(data)


if __name__ == '__main__':
    test()
