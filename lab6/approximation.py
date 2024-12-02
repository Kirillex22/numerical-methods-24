import numpy as np
import matplotlib.pyplot as plt

class Approximator:

    def show_points(self, x, y):
        plt.scatter(x, y, color='blue')
        plt.grid(True)
        plt.xticks(x)
        plt.yticks(y)
        plt.show()

    def rev_matrix(self, system) -> list:
        solution = np.dot(
            np.linalg.inv(
                np.delete(system, -1, axis=1)
            ),
            system[:, -1])

        return solution.tolist()


    def get_delta(self, func, a, b, c, x, y):
        return np.sqrt(
            np.sum(
                np.power(
                    np.array([func(x, a, b, c) for x in x]) - y,
                    2
                )
            )
        )

    def show_func_with_point(self, func, func_label, a, b, c, x, y):
        plt.scatter(x, y, color='blue')
        plt.grid(True)
        plt.xticks(x)
        plt.yticks(y)
        plt.plot(x, [func(x, a, b, c) for x in x], label = func_label)
        plt.legend()
        plt.show()