import torch


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# Aggregate operations BEGIN
def sum_op(row_select: torch.Tensor, table: torch.Tensor, j: int):
    # TODO the type of j can be erroneous.
    return sum([row_select[i] * table[i][j] for i in range(row_select.size(0))])


def count_op(row_select: torch.Tensor):
    return sum([row_select[i] for i in range(row_select.size(0))])
# Aggregate operations END


# Arithmetic operations BEGIN
def diff_op(scalar_output_past3: torch.Tensor, scalar_output_past1: torch.Tensor):
    return scalar_output_past3.sub(scalar_output_past1)
# Arithmetic operations END


# Comparison operations BEGIN
def g_op(i: int, j: int, table: torch.Tensor, pivot: torch.Tensor):
    return torch.tensor(1.) if table[i][j] > pivot else torch.tensor(0.)


def l_op(i: int, j: int, table: torch.Tensor, pivot: torch.Tensor):
    return torch.tensor(1.) if table[i][j] < pivot else torch.tensor(0.)
# Comparison operations END


# Logic operations BEGIN
def and_op(i: int, row_select_past1: torch.Tensor, row_select_past2: torch.Tensor):
    return torch.min(row_select_past1[i], row_select_past2[i])


def or_op(i: int, row_select_past1: torch.Tensor, row_select_past2: torch.Tensor):
    return torch.max(row_select_past1[i], row_select_past2[i])
# Logic operations END


# Assign Lookup operations BEGIN
def assign_op(i: int, row_select: torch.Tensor):
    return row_select[i]
# Assign Lookup operations END


# Reset operations BEGIN
def reset_op():
    return torch.tensor(1.).to(device)
# Reset operations END


OPERATIONS = [sum_op, count_op, diff_op, g_op, l_op, and_op, or_op, assign_op, reset_op]


# Variables calculators BEGIN
"""
I assume that the array of operations has the following order:
Sum, Count, Difference, Greater, Lesser, And, Or, assign, Reset
0    1      2           3        4       5    6   7       8
"""


def calc_scalar_answer(row_select: torch.Tensor, scalar_output_past3: torch.Tensor, scalar_output_past1: torch.Tensor,
                       table: torch.Tensor, alpha_op: torch.Tensor, alpha_col: torch.Tensor):
    return alpha_op[1] * count_op(row_select) + alpha_op[2] * diff_op(scalar_output_past3, scalar_output_past1) + \
        alpha_op[0] * sum([alpha_col[j] * sum_op(row_select, table, j) for j in range(table.size(1))])


# TODO: figure out how this assignment actually works. It does not appear to select which column from the table to be
#  assigned, only the row.
def calc_lookup_answer(i: int, j: int, row_select: torch.Tensor, alpha_col: torch.Tensor, alpha_op: torch.Tensor):
    if i == 0:
        return torch.tensor(0)
    return alpha_col[j] * alpha_op[7] * assign_op(i, row_select)


def calc_row_select(i: int, row_select_past1: torch.Tensor, row_select_past2: torch.Tensor, table: torch.Tensor,
                    l_pivot: torch.Tensor, g_pivot: torch.Tensor, alpha_op: torch.Tensor, alpha_col: torch.Tensor):
    return alpha_op[5] * and_op(i, row_select_past1, row_select_past2) + \
        alpha_op[6] * or_op(i, row_select_past1, row_select_past2) + \
        alpha_op[8] * reset_op() + \
        sum([alpha_col[j] * (alpha_op[3] * g_op(i, j, table, g_pivot) + alpha_op[4] * l_op(i, j, table, l_pivot))
            for j in range(table.size(1))])
# Variables calculators END
