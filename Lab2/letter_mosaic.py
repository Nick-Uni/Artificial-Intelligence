import numpy as np
import random

BOARD = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 1, 0]])


class Checker():
    def check(self, board):
        if self.equal(board) is False or self.consecutive(board) is False:
            return False
        else:
            return True

    def equal(self, board):
        for row in range(1, len(board)):
            if np.array_equal(board[0], board[row]):
                return False
        return True

    def consecutive(self, board):
        sums = [0, 3]
        for i in range(len(board)):
            for j in range(len(board)):
                try:
                    if board[i][j] + board[i-1][j] + board[i+1][j] in sums:
                        return False
                    if board[i][j] + board[i][j-1] + board[i][j+1] in sums:
                        return False
                except IndexError:
                    pass
        return True


if __name__ == "__main__":
    checker = Checker()
    result = []
    for i in range(100):
        matrix = np.random.randint(2, size=(4, 4))
        if checker.check(matrix) is True:
            result.append(matrix)
    print(result)
