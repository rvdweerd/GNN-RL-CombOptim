import itertools

from copy import deepcopy
from dataclasses import dataclass
from typing import List

import clingo
from pysat.formula import CNF
from pysat.solvers import MinisatGH


@dataclass
class TestInput:
    k: int
    sudoku: List[List[int]]
    num_solutions: int


test_inputs = [
    TestInput(
        k=3,
        sudoku=[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 0, 0, 8, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0, 2, 7],
                [0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 9, 0, 0],
                [0, 0, 0, 0, 5, 6, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        num_solutions=1
    ),
    TestInput(
        k=3,
        sudoku=[[9, 5, 6, 1, 2, 4, 0, 0, 0],
                [3, 7, 8, 9, 5, 6, 0, 0, 0],
                [1, 2, 4, 3, 7, 8, 0, 0, 0],
                [8, 9, 5, 6, 1, 2, 0, 0, 0],
                [4, 3, 7, 8, 9, 5, 0, 0, 0],
                [6, 1, 2, 4, 3, 0, 0, 0, 0],
                [7, 8, 9, 5, 6, 0, 0, 0, 0],
                [2, 4, 3, 7, 8, 0, 0, 0, 0],
                [5, 6, 1, 2, 4, 0, 0, 0, 0]],
        num_solutions=1
    ),
    TestInput(
        k=3,
        sudoku=[[9, 5, 6, 0, 0, 4, 0, 0, 0],
                [3, 7, 8, 9, 5, 6, 0, 0, 0],
                [0, 0, 4, 3, 7, 8, 0, 0, 0],
                [8, 9, 5, 6, 0, 0, 0, 0, 0],
                [4, 3, 7, 8, 9, 5, 0, 0, 0],
                [6, 0, 0, 4, 3, 0, 0, 0, 0],
                [7, 8, 9, 5, 6, 0, 0, 0, 0],
                [0, 4, 3, 7, 8, 0, 0, 0, 0],
                [5, 6, 0, 0, 4, 0, 0, 0, 0]],
        num_solutions=2
    ),
    TestInput(
        k=3,
        sudoku=[[9, 0, 0, 0, 0, 4, 0, 0, 0],
                [3, 0, 0, 0, 0, 6, 0, 0, 0],
                [1, 0, 0, 0, 0, 8, 0, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 0, 0, 0, 0],
                [6, 0, 0, 0, 0, 0, 0, 0, 0],
                [7, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 0, 0, 0, 0, 0, 0, 0, 0]],
        num_solutions=1
    ),
    TestInput(
        k=4,
        sudoku=[[3, 1, 15, 13, 7, 16, 5, 12, 4, 2, 14, 10, 8, 9, 6, 0],
                [6, 4, 5, 8, 9, 11, 2, 10, 16, 15, 7, 3, 1, 14, 13, 0],
                [10, 7, 14, 2, 15, 6, 1, 13, 9, 11, 8, 12, 4, 16, 3, 0],
                [12, 16, 9, 11, 3, 14, 4, 8, 6, 5, 1, 13, 2, 15, 7, 0],
                [4, 2, 10, 1, 16, 9, 12, 0, 3, 7, 15, 14, 5, 6, 8, 0],
                [9, 8, 3, 6, 4, 5, 13, 0, 1, 16, 12, 11, 7, 10, 2, 0],
                [15, 13, 12, 14, 2, 8, 7, 0, 5, 10, 4, 9, 3, 1, 11, 0],
                [11, 5, 16, 7, 10, 1, 14, 0, 2, 8, 13, 6, 12, 4, 9, 0],
                [8, 10, 2, 3, 11, 4, 15, 0, 12, 1, 5, 7, 16, 13, 14, 0],
                [16, 9, 13, 15, 8, 12, 6, 0, 14, 3, 10, 2, 11, 5, 1, 0],
                [7, 14, 11, 0, 1, 2, 10, 0, 15, 13, 6, 4, 9, 8, 0, 0],
                [1, 12, 6, 0, 14, 13, 3, 0, 8, 9, 11, 16, 15, 2, 0, 0],
                [13, 3, 7, 0, 12, 15, 9, 0, 10, 14, 2, 1, 6, 11, 0, 0],
                [14, 0, 1, 0, 5, 7, 16, 0, 11, 6, 3, 8, 13, 12, 0, 0],
                [0, 0, 4, 0, 6, 3, 8, 0, 13, 12, 16, 5, 10, 7, 0, 0],
                [5, 6, 8, 12, 13, 10, 11, 1, 7, 4, 9, 15, 14, 3, 0, 2]],
        num_solutions=1
    )
]


def check_solution(sudoku, k, solution):
    """
    Checks if a given solution for a given BombKnightSudoku puzzle is correct.
    """

    # Check if each row in the solution has different values
    for row in solution:
        if set(row) != set(range(1, k**2+1)):
            return False
    # Check if each row in the solution has different values
    for col in range(0, k**2):
        if {solution[row][col]
                for row in range(0, k**2)
           } != set(range(1, k**2+1)):
            return False
    # Check the 'bomb' constraints
    for row1, col1 in itertools.product(range(0, k**2), repeat=2):
        for row_add, col_add in itertools.product([-1, 0, 1], repeat=2):
            if row_add != 0 or col_add != 0:
                row2 = row1 + row_add
                col2 = col1 + col_add
                if 0 <= row2 < k**2 and 0 <= col2 < k**2:
                    if solution[row1][col1] == solution[row2][col2]:
                        return False
    # Check the 'knight' constraints
    for row1, col1 in itertools.product(range(0, k**2), repeat=2):
        for row_add, col_add in [(1, 2), (1, -2), (-1, 2), (-1, -2),
                                 (2, 1), (-2, 1), (2, -1), (-2, -1)]:
            if row_add != 0 or col_add != 0:
                row2 = row1 + row_add
                col2 = col1 + col_add
                if 0 <= row2 < k**2 and 0 <= col2 < k**2:
                    if solution[row1][col1] == solution[row2][col2]:
                        return False
    # Check if each block in the solution has different values
    for block_row, block_col in itertools.product(range(0, k), repeat=2):
        if {solution[block_row*k + inner_row][block_col*k + inner_col]
                for inner_row, inner_col
                in itertools.product(range(0, k), repeat=2)
           } != set(range(1, k**2+1)):
            return False
    # Check if the solution matches the input
    for row, col in itertools.product(range(0, k**2), repeat=2):
        if sudoku[row][col] != 0 and sudoku[row][col] != solution[row][col]:
            return False
    # If all checks passed, return True
    return True


def check_num_solutions(sudoku, k, num_solutions, solver):
    """
    Checks if a given solving algorithm produces the right number of correct
    solutions for a given BombKnightSudoku puzzle.
    """

    # Iterate over num_solutions+1 solutions, check if each is correct,
    # and add their string representations to a set
    solution_set = set()
    for solution in itertools.islice(solver(sudoku, k), num_solutions+1):
        if not check_solution(sudoku, k, solution):
            return False
        solution_set.add(pretty_repr(solution, k))

    # Return whether the set contains exactly the right amount of solutions
    return len(solution_set) == num_solutions

def pretty_repr(sudoku, k):
    """
    Produces a pretty representation of a sodoku or solution.
    """

    repr_sudoku = ""
    numwidth = len(str(k**2))
    def pretty_line(k):
        return "." + ".".join(["-"*((numwidth+1)*k+1)]*k) + ".\n"

    # Add a line separator at the beginning
    repr_sudoku += pretty_line(k)
    # Go through all rows of the sudoku
    for rownum in range(0, k**2):
        # Add a row of the sudoku
        repr_sudoku += "| "
        for outer_col in range(0, k):
            for inner_col in range(0, k):
                if sudoku[rownum][outer_col*k+inner_col] != 0:
                    repr_sudoku += str(
                        sudoku[rownum][outer_col*k+inner_col]
                    ).zfill(numwidth) + " "
                else:
                    repr_sudoku += " "*numwidth + " "
            repr_sudoku += "| "
        repr_sudoku += "\n"
        # Add a line separator after every k'th row
        if (rownum+1) % k == 0:
            repr_sudoku += pretty_line(k)
    # Return the constructed string (without trailing '\n')
    return repr_sudoku[:-1]
    