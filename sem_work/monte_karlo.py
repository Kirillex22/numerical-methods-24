import numpy as np
import math
import random

class MonteCarlo:

    def __init__(self):
        self.rounder = lambda x: round(x, 3)
        self.checker = lambda x, interval: interval[0] <= x < interval[1]
        self.logging = False
        self.stats = {}
        self.points = []


    def generator(self, n: int) -> np.array:
        return np.random.uniform(0, 1, n)


    def integrate(self, func, n: int, ranges: list):
        points, R = [], []
        for curr_range in ranges:
            a, b = curr_range
            R.append(b-a)
            points.append((b-a)*self.generator(n) + a)

        points = np.array(points).T
        c: float = np.prod(np.array(R))/n

        return self.rounder(
            c*sum(func(*point) for point in points)
        )


    def solve_lin_sys(self, a: np.array, b: np.array, N: int) -> np.array:
        n = len(a)
        x = np.zeros(n)
        num_tracks = np.zeros(n)
        V = np.sign(a)
        p = a*V
        transition_matrix = np.zeros((n+1, n+1))
        transition_matrix[:n, :n] = p
        transition_matrix[n, n] = 1
        for i in range(n):
            transition_matrix[i, n] = 1 - np.sum(p[i])

        self.stats['V'] = V
        self.stats['p'] = p
        self.stats['tm'] = transition_matrix
        self.stats['intervals'] = []

        for i in range(n):
            self.stats[i] = []

            xis = [random.uniform(0, 1) for _ in range(N)]
            intervals = [
                (
                    self.rounder(np.sum(transition_matrix[i, :j])),
                    self.rounder(np.sum(transition_matrix[i, :j + 1]))
                )
                for j in range(n + 1)
            ]
            self.stats['intervals'].append(", ".join([f"S{l+1}: {intervals[l]}" for l in range(len(intervals))]))
            transitions = [i]
            current_xis = []
            for xi in xis:
                transition = next(i for i, interval in enumerate(intervals) if self.checker(xi, interval))
                transitions.append(transition)
                current_xis.append(self.rounder(xi))
                if transition == n:
                    X = sum([
                        math.prod(
                            [b[transitions[j]]] +
                            [V[transitions[k],transitions[k + 1]] for k in range(j)]
                        )
                        for j in range(len(transitions) - 1)
                    ])
                    x[i] += X
                    num_tracks[i] += 1

                    self.stats[i].append([
                        int(num_tracks[i]),
                        ", ".join(map(str, current_xis)),
                        " -> ".join(map(str, map(lambda x: x+1, transitions))),
                        str(self.rounder(X))
                    ])

                    transitions = [i]
                    current_xis = []

            x[i] /= num_tracks[i]

        return x


    def calculate_area(self, funcs: list, bounds: list, n: int):
        a, b, c, d = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
        rect_area = (b-a)*(d-c)
        x = [random.uniform(a, b) for _ in range(n)]
        y = [random.uniform(c, d) for _ in range(n)]
        k = 0
        for x, y in zip(x, y):
            if math.prod([func(x, y) for func in funcs]):
                k+=1

        return rect_area*k/n