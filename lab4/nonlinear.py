import numpy as np
import math

class NonlinearSolver:

    def __init__(self):
        self.system = None
        self.derivatives = None
        self.dim = None
        self.eps = None

    def load_system(self, system, derivatives):
        self.system = system
        self.derivatives = derivatives
        self.dim = len(system)

    def get_m(self, start_point) -> list[float]:
        x, y = start_point
        M = []
        for i in range(-1000, 1000):
            if i == 0:
                continue
            dx = 1 - self.derivatives[0][0](x, y)/i
            dy = -self.derivatives[0][1](x, y)/i
            if abs(dx) + abs(dy) < 1:
                M.append(i)
                break

        for i in range(-1000, 1000):
            if i == 0:
                continue
            dx = -self.derivatives[1][0](x, y)/i
            dy = 1 - self.derivatives[1][1](x, y)/i
            if abs(dx) + abs(dy) < 1:
                M.append(i)
                break

        return M

    def jacobi(self) -> np.array:
        current_x = np.array([0 for i in range(self.dim)]).astype(float)
        last_x = np.array([1 for i in range(self.dim)]).astype(float)
        M = self.get_m(current_x.tolist())
        count = 0

        while np.max(np.abs(current_x - last_x)) > self.eps:
            last_x = current_x.copy()
            for i in range(self.dim):
                current_x[i] -= self.system[i](*last_x)/M[i]
            count += 1

        return current_x.tolist(), count


    def zeidel(self) -> np.array:
        current_x = np.array([0 for i in range(self.dim)]).astype(float)
        last_x = np.array([1 for i in range(self.dim)]).astype(float)
        M = self.get_m(current_x.tolist())
        count = 0
        while np.max(np.abs(current_x - last_x)) > self.eps:
            last_x = current_x.copy()
            iter_step = 0
            for i in range(self.dim):
                iter_input = [last_x[i] if i > iter_step else current_x[i] for i in range(self.dim)]
                current_x[i] -= self.system[i](*iter_input) / M[i]
                iter_step += 1
            count += 1

        return current_x.tolist(), count

    def rev_matrix(self, A, b) -> np.array:
        return np.dot(np.linalg.inv(A),b)

    def newton(self, current_x) -> np.array:
        count = 0
        dx = np.array([1, 1]).astype(float)

        while np.max(np.abs(dx)) > self.eps:
            A, b = [], []
            for eq, deq in zip(self.system, self.derivatives):
                A.append([deq[i](*current_x) for i in range(self.dim)])
                b.append(-eq(*current_x))
            A, b = np.array(A).astype(float), np.array(b).astype(float)
            dx = self.rev_matrix(A, b)
            current_x += dx
            count += 1

        return current_x.tolist(), count