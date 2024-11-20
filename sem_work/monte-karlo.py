import numpy as np
import math

class MonteCarlo:

    def __init__(self):
        self.rounder = lambda x: round(x, 3)
        self.checker = lambda x, interval: True if interval[0] <= x < interval[1] else False


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
            transition_matrix[n, i] = 0

        for i in range(n):
            xis = np.random.uniform(0, 1, N)
            intervals = [
                (
                    np.sum(transition_matrix[i, :j]),
                    np.sum(transition_matrix[i, :j + 1])
                )
                for j in range(n + 1)
            ]
            transitions = [i]
            for xi in xis:
                transition = [i for i, interval in enumerate(intervals) if self.checker(xi, interval)][0]
                transitions.append(transition)
                if transition == n:
                    x[i] += sum([
                        math.prod(
                            [b[transitions[j]]] +
                            [V[transitions[k],transitions[k + 1]] for k in range(j)]
                        )
                        for j in range(len(transitions) - 1)
                    ])
                    transitions.clear()
                    transitions.append(i)
                    num_tracks[i] += 1

            x[i] /= num_tracks[i]

        return x


mc = MonteCarlo()
func = lambda x, y, z: x + y*z
ranges = [
    [0, 1],
    [0, 2],
    [0, 3]
]

print(mc.integrate(func, 1000000, ranges))

a = [
    [0.4, -0.1],
    [-0.2, 0.5]
]
b = [0.5, 0.6]

print(mc.solve_lin_sys(a, b, 100))

