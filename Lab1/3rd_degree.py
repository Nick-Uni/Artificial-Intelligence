import matplotlib.pyplot as plt
import numpy as np
import random as rd
import time

POINTS = [(-1, 3, 1), (-6, 6, 0), (7, -5, 0), (0, -7, 0), (-1, -2, 1), (-4, 3, 0), (6, 0, 0), (2, -4, 0), (-0.5, -1, 1),
           (-5, -5, 0), (3, 9, 0), (-4, -4, 0), (-3, 1, 1), (-8, 2, 0), (4, 6, 0), (3, -3, 0), (1, 4, 1), (-7, 10, 0),
           (8, 0, 0), (3, -7, 0), (0, 5, 1), (-5, -3, 0), (4, 0, 0), (-3, -3, 0), (-2, 6, 1), (-8, 9, 0), (7, 2, 0),
           (2, -8, 0), (1, -1, 1), (-6, -3, 0), (4, 10, 0), (-1.4, -3.4, 0), (0, 8, 1), (-4, -3, 0), (6, 8, 0), (0, -6, 0),
           (-1.5, 7, 1), (-8, -1, 0), (7, 6, 0), (-2, -3, 0), (-0, -3, 1), (-4, 9, 0), (8, -6, 0), (-3, -4, 0), (0.5, -1, 1),
           (-8, -3, 0), (5, -6, 0), (1.2, -3.2, 0), (-1, 1, 1), (-6, -7, 0), (5, -7, 0), (1, -7, 0), (2, 4, 1), (-6, 5, 0),
           (4, 3, 0), (2, -6, 0), (0, 2, 1), (-6, -1, 0), (6, 4, 0), (-1, -6, 0), (0, 1, 1), (-3, 10, 0), (4, 5, 0),
           (2, -7, 0), (2, 6, 1), (-3, -7, 0), (8, 8, 0), (-3, -8, 0), (2, 8, 1), (-5, 7, 0), (3, 4, 0), (-2, -5, 0),
           (0, -1, 1), (-4, 6, 0), (8, -3, 0), (-3, -5, 0), (1, 2, 1), (-8, 7, 0), (8, -1, 0), (-1, -7, 0), (-2, 7, 1),
           (-7, -5, 0), (7, 8, 0), (-2, -7, 0), (-1, 3.7, 1), (-5, -7, 0), (7, 5, 0), (-4, -7, 0), (1, -2, 1), (-3, 1, 0),
           (3, 6, 0), (3, -6, 0), (-1.2, 0, 1), (-6, -2, 0), (6, -5, 0), (-4, -6, 0), (-2, 4, 1), (-5, 1, 0), (7, -6, 0),
           (2, -3, 0)]


LINES = [(-2, 8, -5, 5), (-9, -7, -4, -9), (3, -10, 1, 0), (1, -7, 6, 10), (2, -5, -6, -1), (1, 9, 6, -6),
         (-3, -2, -5, -3), (6, 3, 5, -3), (-5, -4, 7, 1), (10, 6, 6, -6), (6, -7, -7, 5), (2, 0, 7, 5),
         (-2, 10, -8, 4), (3, 7, -2, -3), (-6, -10, -5, 7), (-3, 6, -3, -10), (4, 4, 3, -2), (10, 2, -6, 9),
         (-8, -2, 3, -10), (10, 2, -10, -8), (6, 3, -3, -1), (3, 4, 7, 0), (2, 0, -1, -1), (4, 0, -4, -10),
         (3, -9, 3, 0), (-7, 0, 9, -9), (-5, -7, 1, 10), (-9, 7, -3, 9), (-4, -5, -9, -6), (-10, -2, -10, 10),
         (9, -7, 3, 6), (-3, -1, 2, 4), (-10, -10, 1, 3), (-6, -3, 4, 6), (0, 8, 10, 7), (1, -7, -5, -3),
         (8, -5, 10, -9), (-6, 4, 8, -7), (-4, 2, -9, -5), (10, -8, -4, -1), (-8, -1, -5, 0), (10, 8, -8, 7),
         (7, 7, 7, -5), (2, -9, 7, -5), (5, 1, -1, 9), (-3, -9, 0, 0), (-5, -6, -7, -9), (-6, 8, 7, 0),
         (-1, 0, 9, -3), (2, -2, -9, 9), (4, 10, -3, 1), (1, 6, -1, 2), (10, -8, 3, 3), (-2, -9, -7, -7),
         (-1, 8, -10, -9), (-10, -7, 9, 7), (-7, -10, 8, -8), (-6, -4, 9, 9), (4, 5, -5, 9), (-1, 0, -6, 5),
         (-7, -4, 0, 1), (-6, -6, 9, 0), (-3, 10, -3, 7), (-1, -1, -10, -1), (4, -5, 2, 1), (5, -5, -2, 6),
         (6, -4, -2, 9), (-7, -3, 8, -10), (-3, -5, -4, -5), (-1, -9, -10, -3), (-1, 8, -1, 9), (4, 6, -6, -7),
         (10, -1, -10, 9), (-10, -9, 2, -5), (9, -5, -9, -3), (0, -7, -6, 8), (1, -6, 3, -8), (-2, 2, 6, -2),
         (10, 10, 1, 4), (-8, 5, -1, 8), (5, 4, -4, -1), (-8, -5, 3, -8), (-4, -4, 9, -7), (4, 7, -3, -2),
         (7, -3, -4, 9), (-9, 10, -3, -6), (5, 8, -5, -10), (8, -9, -1, 8), (-2, 9, 5, -1), (-5, -1, 9, -3),
         (-5, -1, 3, 4), (4, -3, -6, 4), (-5, 5, 9, -8), (2, 9, 4, 8), (-10, 8, 1, 4), (-9, 10, 7, 1), (-9, 7, -8, 0),
         (5, -9, -7, -1), (0, -1, -5, -2), (-1, -8, 8, -6)]



class GeneticAlgorithm:

    def fitness(self, points, lines):
        result = {}
        for line in lines:
            if type(line) is list:
                line = tuple(line)
            correct = 0
            for point in points:
                if point[1] > line[0] * (point[0]-line[3])**2 + line[1] * (point[0]-line[3]) + line[2] and point[2] == 1:
                    correct += 1
                elif point[1] < line[0] * (point[0]-line[3])**2 + line[1] * (point[0]-line[3]) + line[2] and point[2] == 0:
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
        formula = (lambda x: line[0] * (x-line[3])**2 + line[1] * (x-line[3]) + line[2])
        x = np.linspace(-10, 10, 1000)
        y = formula(x)
        plt.plot(x, y)
        for point in points:
            if point[2] == 1:
                plt.plot([point[0]], [point[1]], 'ro')
            else:
                plt.plot([point[0]], [point[1]], 'bo')
        plt.axis([-10, 10, -10, 10])
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
            a1_bin = bin(list(lines_sorted)[random_1][0])
            b1_bin = bin(list(lines_sorted)[random_1][1])
            c1_bin = bin(list(lines_sorted)[random_1][2])
            h1_bin = bin(list(lines_sorted)[random_1][3])
            raw_param_bin = [a1_bin, b1_bin, c1_bin, h1_bin]
            a2_bin = bin(list(lines_sorted)[random_2][0])
            b2_bin = bin(list(lines_sorted)[random_2][1])
            c2_bin = bin(list(lines_sorted)[random_2][2])
            h2_bin = bin(list(lines_sorted)[random_2][3])
            raw_param_bin.extend([a2_bin, b2_bin, c2_bin, h2_bin])
            param_bin = []
            for param in raw_param_bin:
                splited = param.split('0b')
                param_bin.append([splited[1], splited[0], len(splited[1])])
            for i in range(0, 4):
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
    lines = lines[:50]
    start = time.time()
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
    finish = time.time()
    print(finish - start)
    print(best, '{} %, {} generation'.format(percent, solution))
    genetic.plot(POINTS, best[0][0])
    return percent, solution

if __name__ == '__main__':
    solution_generation = []
    result_percent = []
    for i in range(10):
        percent, generation = start(LINES)
        result_percent.append(percent)
        solution_generation.append(generation)
    average_percent = int(sum(result_percent)/len(result_percent))
    average_generation = int(sum(solution_generation) / len(solution_generation))
    print('Average generation of solution: {}, Average percent of solution: {}, Best result {}'.format(average_generation, average_percent, max(result_percent)))