from collections import OrderedDict
from sortedcontainers import SortedDict
import matplotlib.pyplot as plt
import numpy as np


class GenetiveAlgorithm:

    def fitness(self, points, lines):
        result = {}
        for line in lines:
            correct = 0
            for point in points:
                if point[1] > line[0] * point[0] + line[1] and point[2] == 1:
                    correct += 1
                elif point[1] < line[0] * point[0] + line[1] and point[2] == 0:
                    correct += 1
            result[line] = correct
        return SortedDict(result)
        # return result

    def run(self, points, lines):
        result = self.fitness(points, lines)
        print(result)
        self.plot(points, result.keys()[-1])

    def plot(self, points, line):
        self.graph(lambda x: x * line[0] + line[1], range(0, 11))
        for point in points:
            if point[2] == 1:
                plt.plot([point[0]], [point[1]], 'ro')
            else:
                plt.plot([point[0]], [point[1]], 'bo')
        plt.show()

    def graph(self, formula, x_range):
        x = np.array(x_range)
        y = formula(x)
        plt.plot(x, y)

if __name__ == '__main__':
    points = [(1, 1, 1), (1, 3, 1), (1, 4, 1), (2, 3, 1), (2, 5, 1), (2, 7, 1), (3, 2, 1), (3, 5, 1), (4.5, 4, 1),
              (5, 4, 1), (5, 5, 1), (9, 1, 0), (9, 1, 0), (6, 3, 0), (7, 4, 0), (8, 5, 0)]
    lines = [(1, -1), (-4, -8), (2, -6)]
    genetive = GenetiveAlgorithm()
    genetive.run(points, lines)
