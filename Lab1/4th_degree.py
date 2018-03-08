import matplotlib.pyplot as plt
import numpy as np
import random as rd

POINTS = [(1, 1, 1), (1, 3, 1), (1, 4, 1), (2, 3, 1), (2, 5, 1), (2, 7, 1), (3, 2, 1), (3, 5, 1), (4.5, 4, 1),
          (5, 8, 1), (5, 9, 1), (9, 1, 0), (9, 1, 0), (6, 3, 0), (7, 4, 0), (8, 5, 0), (0, -1, 1), (1, -2, 1),
          (6, 0, 0), (8, 6, 0), (6, 4, 0), (7, 6, 0), (6.5, 4, 0), (6, -1, 0), (7, 6, 0), (1.5, 3, 1), (2.5, 4, 1),
          (4, -2, 1), (6, 10, 0), (7, 7, 0), (4, 8, 1), (7, 11, 0), (2.5, -2.5, 1), (3, 0, 1), (4.9, 2.4, 0), (6, 4, 0),
          (4.8, -5, 0), (3, -6, 0), (6, -2.5, 0), (6, 5, 0), (5, -2.5, 0), (1, 0, 1), (2, 1, 1), (2, 6, 1), (0, -2.5, 1),
          (3, -7.5, 0), (5.1, 0.1, 0)]

LINES = [(20, 8, 8, 12, -10), (-8, -15, -12, 0, -18), (10, 0, -15, -11, -6), (-6, 16, 19, -20, 14),
         (11, 18, 3, 17, -14), (-20, -14, -6, 20, -17), (-19, 9, 2, -11, -7), (14, 7, -12, -9, -1),
         (14, -20, 2, -8, 8), (-9, -18, 1, 5, 13), (-18, -9, -15, -17, 5), (2, -13, 9, 16, 11),
         (4, 0, 0, 16, -3), (-8, -12, -13, 10, -19), (6, -16, 5, -18, -2), (-4, -19, -8, 5, 3),
        (-15, 7, 16, 17, -19), (1, 13, -1, -19, -18), (-1, 9, 16, 7, -14), (-8, -17, 12, 12, -1),
        (-11, -20, -16, 19, -11), (10, 8, -16, -13, -17), (9, 14, 15, -17, -2), (-10, -15, -6, -17, 7),
        (19, -5, -16, -17, -5), (-17, 9, 12, -14, -15), (14, 2, 13, -3, -3), (8, -3, -16, -18, 5),
        (1, -12, 10, 12, 16), (-6, -4, 20, 7, 6), (13, 13, -19, 11, 19), (7, -2, 12, -9, -1),
        (-16, -6, 20, 3, 14), (5, -18, -16, 13, 19), (-14, -19, 17, -11, 19), (16, -2, -12, 9, 16),
        (-18, 7, -2, -11, -5), (16, 1, 16, 18, -17), (-9, 11, -16, 14, -14), (-8, -17, 4, 17, 9),
        (-15, -19, 10, -18, -13), (-6, -4, 2, -13, -3), (-18, 18, -13, -15, 14), (1, -13, -1, 12, 6),
        (-20, -19, -5, -14, -7), (5, -19, -13, 7, -18), (12, 7, 13, -8, -17), (-4, 9, -3, -15, 9),
        (0, -10, 3, -16, -14), (19, 7, -2, -14, 14), (12, 0, -9, -20, -11), (-16, -1, -6, -12, 17),
        (6, -8, 1, -17, -2), (10, -20, -6, 3, 18), (-17, 10, -7, 7, 10), (5, 20, 17, -9, 15),
        (20, -8, -13, 12, -12), (-7, 13, 10, 12, -3), (-5, 19, 20, -14, -17), (-5, -4, -2, -20, -2),
        (6, -11, -20, 20, -16), (-5, -14, 7, 3, -14), (19, -13, -19, 9, -6), (-8, 18, -9, 3, 15),
        (-20, -12, 17, -14, -3), (-20, 18, 11, 19, -11), (3, -6, -1, 1, -3),
        (-8, 2, 10, -5, 4), (-1, -11, -19, 8, 18), (12, 7, -15, -9, 8), (5, -18, -4, -20, -12),
        (-19, -12, 14, -18, -9), (-1, -6, 10, -2, 16), (-5, 20, -14, -14, 13), (-7, -1, 7, -8, 8),
        (7, 19, -13, -4, -6), (17, -15, 15, -16, 11), (0, 7, 14, -13, 9), (4, 8, -20, -8, -10),
        (0, -4, 2, -5, -1), (-1, 5, -16, 5, -15), (4, 14, 14, 1, 5), (7, -20, -18, -8, 6),
        (6, -13, -15, -6, 18), (-1, 9, -17, -11, 14), (-16, 0, 4, -1, -10), (-6, 17, -4, 13, 4),
        (-6, 9, -2, 11, 11), (-18, -4, 18, 20, -3), (13, 2, -3, 10, -13), (10, 17, -19, -6, 5),
        (-20, -17, 12, -13, -6), (12, -18, 18, -19, 19), (1, -16, 3, 1, -17), (14, -12, 9, 2, -17),
        (19, 13, -4, -17, 11), (-4, -19, -13, 4, -20), (14, -7, 0, -7, 20), (1, 2, 3, 4, 5), (10, 4, 2, -6, 1)]


class GeneticAlgorithm:

    def fitness(self, points, lines):
        result = {}
        for line in lines:
            if type(line) is list:
                line = tuple(line)
            correct = 0
            for point in points:
                if point[1] > line[0] * (point[0]-line[4])**3 + line[1] * (point[0]-line[4])**2 + line[2] * (point[0]-line[2]) + line[3] and point[2] == 1:
                    correct += 1
                elif point[1] < line[0] * (point[0]-line[4])**3 + line[1] * (point[0]-line[4])**2 + line[2] * (point[0]-line[2]) + line[3] and point[2] == 0:
                    correct += 1
            result[line] = correct

        best_result = max(result.values())
        best = list(filter(lambda x: x[1] == best_result, result.items()))
        best_line = []
        best_line.append(best[0])
        for i in best:
            del result[i[0]]
        best_result = max(result.values())
        lst = list(filter(lambda x: x[1] == best_result, result.items()))
        best_line.append(lst[0])
        return best_line

    def run(self, points, lines):
        result = self.fitness(points, lines)
        self.plot_4th_degree(points, result[0][0])

    def graph_4th_degree(self, formula, x_range):
        x = np.array(x_range)
        y = formula(x)
        plt.plot(x, y)

    def plot_4th_degree(self, points, line):
        self.graph_4th_degree(lambda x: line[0] * (x-line[4])**3 + line[1] * (x-line[4])**2 + line[2] * (x-line[2]) + line[3], range(-10, 10))
        for point in points:
            if point[2] == 1:
                plt.plot([point[0]], [point[1]], 'ro')
            else:
                plt.plot([point[0]], [point[1]], 'bo')
        plt.axis([-10, 10, -10, 10])
        plt.show()

    def plot(self, points, line):
        self.graph(lambda x: x * line[0] + line[1], range(0, 11))
        for point in points:
            if point[2] == 1:
                plt.plot([point[0]], [point[1]], 'ro')
            else:
                plt.plot([point[0]], [point[1]], 'bo')
        plt.axis([-5, 15, -10, 15])
        plt.show()

    def graph(self, formula, x_range):
        x = np.array(x_range)
        y = formula(x)
        plt.plot(x, y)

    def mutation(self, number, mutation_variance):
        number_list = []
        result = ''
        for bit in number:
            mutation_rand = rd.randint(0, 100)
            if mutation_rand >= mutation_variance:
                if bit == '0':
                    number_list.append('1')
                else:
                    number_list.append('0')
            else:
                number_list.append(bit)
        for i in number_list:
            result = result + i
        return result

    def mutatation_plus_minus(self, x1, x2, mutation_variance):
        if x1 == '-' and x2 == '-':
            plus_minus = ''
        elif x1 == '' and x2 == '-':
            plus_minus = '-'
        elif x1 == '-' and x2 == '':
            plus_minus = '-'
        else:
            plus_minus = ''
        mutation_rand = rd.randint(0, 100)
        if mutation_rand >= mutation_variance:
            if plus_minus == '-':
                plus_minus = ''
            else:
                plus_minus = '-'
        return plus_minus

    def crossover(self, lines_sorted):
        result = []
        mutation_variance = 50
        values = []
        for line in range(int(len(lines_sorted)/2)):
            random_1 = rd.randint(0, 97)
            random_2 = rd.randint(0, 97)
            while random_1 in values:
                random_1 = rd.randint(0, len(lines_sorted)-1)
            while random_2 in values:
                random_2 = rd.randint(0, len(lines_sorted)-1)
            values.append(random_1)
            values.append(random_2)
            a1_bin = bin(list(lines_sorted)[random_1][0])
            b1_bin = bin(list(lines_sorted)[random_1][1])
            c1_bin = bin(list(lines_sorted)[random_1][2])
            d1_bin = bin(list(lines_sorted)[random_1][3])
            h1_bin = bin(list(lines_sorted)[random_1][4])
            raw_param_bin = [a1_bin, b1_bin, c1_bin, d1_bin, h1_bin]
            a2_bin = bin(list(lines_sorted)[random_2][0])
            b2_bin = bin(list(lines_sorted)[random_2][1])
            c2_bin = bin(list(lines_sorted)[random_2][2])
            d2_bin = bin(list(lines_sorted)[random_2][3])
            h2_bin = bin(list(lines_sorted)[random_2][4])
            raw_param_bin.extend([a2_bin, b2_bin, c2_bin, d2_bin, h2_bin])
            param_bin = []
            for param in raw_param_bin:
                splited = param.split('0b')
                param_bin.append([splited[1], splited[0], len(splited[1])])
            for i in range(0, 5):
                if param_bin[i][2] > param_bin[i+2][2]:
                    param_bin[i + 2][0] = '0' * (param_bin[i][2] - param_bin[i + 2][2]) + param_bin[i + 2][0]
                    param_bin[i+2][2] = len(param_bin[i+2][0])
                else:
                    param_bin[i][0] = '0' * (param_bin[i+2][2] - param_bin[i][2]) + param_bin[i][0]
                    param_bin[i][2] = len(param_bin[i][0])
                cros_int = int(0.5 * param_bin[i][2])
                plus_minus = self.mutatation_plus_minus(param_bin[i][1], param_bin[i+2][1], mutation_variance)
                a_bin = param_bin[i][0][:cros_int] + param_bin[i + 2][0][cros_int:]
                a_bin = self.mutation(a_bin, mutation_variance)
                a_bin = plus_minus + a_bin
                b_bin = param_bin[i + 2][0][:cros_int] + param_bin[i][0][cros_int:]
                b_bin = self.mutation(b_bin, mutation_variance)
                b_bin = plus_minus + b_bin
                a = int(a_bin, 2)
                b = int(b_bin, 2)

                if i == 0:
                    result.append([a])
                    result.append([b])
                else:
                    result[-1].append(b)
                    result[-2].append(a)
        return result


def start(lines):
    lines = lines.copy()
    genetic = GeneticAlgorithm()
    best = genetic.fitness(POINTS, lines)
    for line in best:
        if line[0] in lines:
            lines.remove(line[0])
        elif list(line[0]) in lines:
            lines.remove(list(line[0]))
    # print(best, '{} %'.format(best[0][1]/len(POINTS)*100))
    # genetic.run(POINTS, lines)
    for i in range(1, 101):
        lines = genetic.crossover(lines)
        for j in range(2):
            lines.append(best[j][0])
        best_new = genetic.fitness(POINTS, lines)
        if best_new[0][1] > best[0][1]:
            best = best_new
            solution = i

        for line in best:
            if line[0] in lines:
                lines.remove(line[0])
            elif list(line[0]) in lines:
                lines.remove(list(line[0]))
        if best[0][1]/len(POINTS) == 1:
            break
    percent = best[0][1] / len(POINTS) * 100
    print(best, '{} %, {} generation'.format(percent, solution))
    genetic.run(POINTS, lines)
    return percent, solution

if __name__ == '__main__':
    solution_generation = []
    result_percent = []
    for i in range(1000):
        percent, generation = start(LINES)
        result_percent.append(percent)
        solution_generation.append(generation)
    print(result_percent)
    # average_percent = int(sum(result_percent)/len(result_percent))
    # average_generation = int(sum(solution_generation) / len(solution_generation))
    # print('Average generation of solution: {}, Average percent of solution: {}'.format(average_generation, average_percent))
    # # genetic = GeneticAlgorithm()
    # # genetic.plot_4th_degree(POINTS, LINES_4TH_DEGREE[0])
    # result = []
    # for i in range(100):
    #     result.append((rd.randint(-20, 20), rd.randint(-20, 20), rd.randint(-20, 20), rd.randint(-20, 20),
    #                    rd.randint(-20, 20)))
    # print(result)
    # genetic = GeneticAlgorithm()
    # fitness = genetic.fitness(POINTS, LINES)
    # print(fitness)
    # genetic.plot_4th_degree(POINTS, fitness[0][0])