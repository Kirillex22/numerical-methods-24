import numpy as np

class MatrixSolver:

    def __init__(self):
        self.matrix: np.array = None
        self.dim = None


    def load_matrix(self, matrix: np.array):
        self.matrix = matrix
        self.dim = len(matrix)


    def get_own_nums(self):
        dets: list[float] = [np.linalg.det(self.matrix - i*np.eye(self.dim)) for i in range(self.dim)]

        raw_b: list[list[int]] = []

        for i in range(1, self.dim):
            raw_b.append([i**j for j in reversed(range(1, self.dim))])

        B: np.array = np.array(raw_b)

        rev_b = np.linalg.inv(B)

        d: list[float] = []

        for i in range(1, self.dim):
            d.append(dets[i] - dets[0] - i**self.dim)

        return np.dot(rev_b, d)


ms = MatrixSolver()
matrix = np.array([
        [1.46, 23.14, -0.78, 1.13],
        [2.31, 1.58, 6.73, 1.61],
        [-0.13, -9.21, 7.41, 1.23],
        [0.96, 1.23, 3.79, 5.46]
    ]).astype(np.float64)

ms.load_matrix(matrix)

print(ms.get_own_nums())
print(np.linalg.eig(matrix)[0])