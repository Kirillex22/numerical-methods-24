import numpy as np

class SlauSolver:

    def __init__(self):
        self.system: np.array = None
        self.dim = None
        self.degree = 6
        self.rounder = lambda x: np.round(x, self.degree)
        self.eps = None


    def load_system(self, system: np.array):
        self.system = system.astype(np.float64)
        self.dim = len(self.system)


    def gauss(self) -> list:
        prep_sys: np.array = self.system
        for i in range(self.dim):
            for k in range(i):
                if prep_sys[i][k] != 0:
                    temp = prep_sys[k] * prep_sys[i][k]/prep_sys[k][k]
                    prep_sys[i] -= temp

        solution: list[float] = [0 for i in range(self.dim)]

        for i in reversed(range(self.dim)):
            s: float = sum([prep_sys[i][j]*solution[j] for j in range(i+1, self.dim)])
            solution[i] = (prep_sys[i][-1] - s)/prep_sys[i][i]

        return self.rounder(solution)


    def rev_matrix(self) -> list:
        solution = np.dot(
            np.linalg.inv(
                np.delete(self.system, -1, axis=1)
            ),
            self.system[:, -1])

        return self.rounder(solution.tolist())


    def zeidel(self, current_x: np.array) -> (list, int):
        A: np.array = np.delete(self.system, -1, axis=1)
        b: np.array = self.system[:, -1]
        previous_x: np.array = np.array([0 for i in range(self.dim)])
        num_iters = 0
        while np.max(np.abs(current_x - previous_x)) > self.eps:
            previous_x = current_x
            temp: np.array = np.array([i for i in current_x])

            for i in range(self.dim):
                sum1 = sum([A[i, j] * temp[j] for j in range(i)])
                sum2 = sum([A[i, j] * current_x[j] for j in range(i + 1, self.dim)])
                temp[i] = (1 / A[i, i]) * (b[i] - sum1 - sum2)

            current_x = temp
            num_iters += 1

        return self.rounder(current_x).tolist(), num_iters


    def run_trough(self):
        A: np.array = np.delete(self.system, -1, axis=1)
        b: np.array = self.system[:, -1]
        u: np.array = np.array([0 for i in range(self.dim)]).astype(np.float64)
        v: np.array = np.array([0 for i in range(self.dim)]).astype(np.float64)
        solution: np.array = np.array([0 for i in range(self.dim)]).astype(np.float64)

        u[0] = -self.system[0, 1] / self.system[0, 0]
        v[0] = self.system[0, -1] / self.system[0, 0]

        for i in range(1, self.dim):
            u[i] = -A[i, (i + 1) % self.dim] / (A[i, i - 1] * u[i - 1] + A[i, i])
            v[i] = (b[i] - A[i, i - 1] * v[i - 1]) / (A[i, i - 1] * u[i - 1] + A[i, i])

        solution[-1] = v[-1]

        for i in reversed(range(self.dim - 1)):
            solution[i] = u[i] * solution[i + 1] + v[i]

        return solution


    def jacobi(self, current_x: np.array) -> (list, int):
        A: np.array = np.delete(self.system, -1, axis=1)
        b: np.array = self.system[:, -1]
        previous_x: np.array = np.array([0 for i in range(self.dim)])
        num_iters = 0
        while np.max(np.abs(current_x - previous_x)) > self.eps:
            previous_x = current_x
            temp: np.array = np.array([0 for i in range(self.dim)])
            for i in range(self.dim):
                sum1 = sum([A[i, j]*current_x[j] for j in range(i)])
                sum2 = sum([A[i, j]*current_x[j] for j in range(i+1, self.dim)])
                temp[i] = (1/A[i,i])*(b[i] - sum1 - sum2)

            current_x = temp
            num_iters += 1

        return self.rounder(current_x).tolist(), num_iters

