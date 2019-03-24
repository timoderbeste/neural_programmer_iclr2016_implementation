"""
This table generator currently generates only the following type of tables with standard answers for the specified questions

- Single column experiment:
    - sum
    - count
    - greater [number] sum
"""

import random


def generate_single_column_table(min_limit: int, max_limit: int):
    num_rows = random.randint(min_limit, max_limit)
    row = []
    for _ in range(num_rows):
        row.append(float(random.randint(min_limit, max_limit)))

    return [row]
