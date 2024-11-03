import numpy as np

class MatrixSolver:

    def __init__(self):
        self.matrix: np.array = None
        self.dim = None


    def load_matrix(self, matrix: np.array):
        self.matrix = matrix
        self.dim = len(matrix)


    def get_own_nums(self) -> np.array:
        dets: list[float] = [np.linalg.det(self.matrix - i*np.eye(self.dim)) for i in range(self.dim)]

        raw_b: list[list[int]] = []

        for i in range(1, self.dim):
            raw_b.append([i**j for j in reversed(range(1, self.dim))])

        B: np.array = np.array(raw_b)
        rev_b = np.linalg.inv(B)

        d = []
        for i in range(1, self.dim):
            d.append(dets[i] - dets[0] - i**self.dim)

        d = np.array(d)
        p = np.array([1] + np.dot(rev_b, d).tolist() + [dets[0]])

        return np.roots(p)


    def rev_matrix(self, A, b) -> np.array:
        return np.dot(np.linalg.inv(A),b)


    def get_own_vectors(self, own_nums: np.array) -> np.array:
        solution = []
        for num in own_nums:
            current_x = np.array([1 for i in range(self.dim)]).T
            system = self.matrix - np.eye(self.dim) * num
            _num = num + 1

            while np.abs(_num - num) > 1e-5:
                current_y = self.rev_matrix(system, current_x)
                current_x = current_y/np.linalg.norm(current_y)
                _num = np.dot(current_x.T, np.dot(self.matrix, current_x))/np.dot(current_x.T, current_x)

            solution.append(current_x)

        return solution