import random
import argparse

import src.data_generators.question_generator as qg
import src.data_generators.table_generator as tg


def generate_single_column_table_dataset(size: int, min_limit: int, max_limit: int):
    generated_tables = []
    generated_data = []

    while len(generated_data) < size:
        table_idx = random.randint(0, len(generated_tables))
        if table_idx < len(generated_tables):
            table = generated_tables[table_idx]
        else:
            table = tg.generate_single_column_table(min_limit, max_limit)
            generated_tables.append(table)

        question_generation_function = \
            qg.QUESTION_GENERATION_FUNCTIONS[random.randint(0, len(qg.QUESTION_GENERATION_FUNCTIONS) - 1)]

        if question_generation_function == qg.generate_single_column_table_sum_all:
            question, answer = question_generation_function(table)
        elif question_generation_function == qg.generate_single_column_table_count_all:
            question, answer = question_generation_function(table)
        # elif question_generation_function == gq.generate_single_column_table_sum_greater:
        else:
            question, answer = question_generation_function(table, min_limit, max_limit)

        generated_data.append((table_idx, question, answer))

    return generated_tables, generated_data

