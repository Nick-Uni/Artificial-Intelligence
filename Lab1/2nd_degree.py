import matplotlib.pyplot as plt
import numpy as np
import random as rd
import time

POINTS = [(1, 1, 1), (1, 3, 1), (1, 4, 1), (2, 3, 1), (2, 5, 1), (2, 7, 1), (3, 2, 1), (3, 5, 1), (4.5, 4, 1),
          (5, 8, 1), (5, 9, 1), (9, 1, 0), (9, 1, 0), (6, 3, 0), (7, 4, 0), (8, 5, 0), (0, -1, 1), (1, -2, 1),
          (6, 0, 0), (8, 6, 0), (6, 4, 0), (7, 6, 0), (6.5, 4, 0), (6, -1, 0), (7, 6, 0), (1.5, 3, 1), (2.5, 4, 1),
          (3, -2, 1), (6, 10, 0), (7, 7, 0), (4, 8, 1), (7, 11, 0), (2.5, -2.5, 1), (3, 0, 1), (4.9, 2.4, 0), (6, 4, 0),
          (4.8, -5, 0), (3, -6, 0), (6, -2.5, 0), (6, 5, 0), (5, -2.5, 0), (1, 0, 1), (2, 1, 1), (2, 6, 1), (0, -2.5, 1),
          (3, -7.5, 0), (5.1, 0.1, 0)]

LINES = [(16, 2), (-7, -18), (19, 10), (-10, -9), (10, -6), (-6, -14), (17, 11), (-9, -7), (-8, -12), (-3, -3),
         (18, -3), (-17, 20), (9, -15), (16, 2), (-5, -15), (-7, 6), (1, 8), (-12, -11), (6, -6), (19, -3),
         (10, -1), (-3, 1), (14, -16), (17, 13), (19, -3), (17, 6), (3, 15), (-9, -9), (15, 2),
         (9, 0), (-3, -2), (-18, -18), (-14, 0), (-7, -15), (-1, -4), (-15, -2), (17, -13), (14, -11), (-4, 11),
         (-17, 20), (14, -1), (0, -11), (-13, 8), (15, 6), (-10, -18), (-6, 11), (20, -3), (16, -2), (14, 8),
         (-20, -5), (0, -4), (-2, 5), (11, -14), (-14, -20), (19, 11), (-3, -5), (15, 16), (17, 4), (14, -10),
         (-13, 19), (-11, -8), (-7, 0), (-10, 3), (-18, -11), (-17, -19), (9, -8), (-17, 9), (8, 19), (-14, 15),
         (11, 12), (-3, -16), (0, 15), (14, 10), (16, 10), (-14, -13), (4, -6), (-3, 19), (-5, 19), (14, -19),
         (-13, -3), (-5, 16), (-4, -11), (4, 19), (-16, 11), (-10, -6), (-6, 2), (18, 17), (11, -6),
         (1, 11),  (-5, -19), (-17, 19), (17, 9), (7, -1), (-3, 9), (10, -11), (4, 5), (40, 30), (-2, 5), (4, 15),
         (6, 1), (5, 4), (10, -14)]


class GeneticAlgorithm:

    def fitness(self, points, lines):
        result = {}
        for line in lines:
            if type(line) is list:
                line = tuple(line)
            correct = 0
            for point in points:
                if point[1] > line[0] * point[0] + line[1] and point[2] == 1:
                    correct += 1
                elif point[1] < line[0] * point[0] + line[1] and point[2] == 0:
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
            random_1 = rd.randint(0, len(lines_sorted)-1)
            random_2 = rd.randint(0, len(lines_sorted)-1)
            while random_1 in values:
                random_1 = rd.randint(0, len(lines_sorted)-1)
            while random_2 in values:
                random_2 = rd.randint(0, len(lines_sorted)-1)
            values.append(random_1)
            values.append(random_2)
            x1_bin = bin(list(lines_sorted)[random_1][0])
            y1_bin = bin(list(lines_sorted)[random_1][1])
            raw_param_bin = [x1_bin, y1_bin]
            x2_bin = bin(list(lines_sorted)[random_2][0])
            y2_bin = bin(list(lines_sorted)[random_2][1])
            raw_param_bin.extend([x2_bin, y2_bin])
            param_bin = []
            for param in raw_param_bin:
                splited = param.split('0b')
                param_bin.append([splited[1], splited[0], len(splited[1])])
            for i in range(0, 2):
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
    start = time.time()
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
            # genetic.run(POINTS, lines)
        for line in best:
            if line[0] in lines:
                lines.remove(line[0])
            elif list(line[0]) in lines:
                lines.remove(list(line[0]))
        if best[0][1]/len(POINTS) == 1:
            break
    percent = best[0][1] / len(POINTS) * 100
    genetic.plot(POINTS, best[0][0])
    finish = time.time()
    print(finish - start)
    print(best, '{} %, {} generation'.format(percent, solution))
    return percent, solution

if __name__ == '__main__':
    solution_generation = []
    result_percent = []
    for i in range(10):
        percent, generation = start(LINES)
        result_percent.append(percent)
        solution_generation.append(generation)
    print(result_percent)
    average_percent = int(sum(result_percent)/len(result_percent))
    average_generation = int(sum(solution_generation) / len(solution_generation))
    print('Average generation of solution: {}, Average percent of solution: {}'.format(average_generation, average_percent))