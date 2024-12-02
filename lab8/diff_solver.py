import enum
from sympy import symbols, Eq
import numpy as np
from prettytable import PrettyTable

class DiffSolver:
    def __init__(self):
        self.r = lambda x: round(x, 3)
        self.f = None

    def make_table(self, data, names):
        _t = PrettyTable()
        _t.field_names = names
        for row in data:
            _t.add_row(list(map(self.r, row)))

        return _t


    def euler_method(self, point, bounds, h):
        x = np.arange(bounds[0], bounds[1] + h, h)
        y = np.zeros(len(x))
        y[0] = point[1]
        for i in range(1, len(x)):
            y[i] = y[i-1] + h*self.f(x[i-1], y[i-1])

        return zip(x, y)


    def mod_euler_method(self, point, bounds, h):
        x = np.arange(bounds[0], bounds[1] + h, h)
        y = np.zeros(len(x))
        y[0] = point[1]
        for i in range(1, len(x)):
            _y = y[i-1] + h*self.f(x[i-1], y[i-1])
            y[i] = y[i-1] + (h/2)*(self.f(x[i-1], y[i-1]) + self.f(x[i], _y))

        return zip(x, y)


    def runge_kutta_method(self, point, bounds, h):
        x = np.arange(bounds[0], bounds[1] + h, h)
        y = np.zeros(len(x))
        y[0] = point[1]
        for i in range(1, len(x)):
            k0 = h*self.f(x[i-1], y[i-1])
            k1 = h*self.f(x[i-1] + h/2, y[i-1] + k0/2)
            k2 = h*self.f(x[i-1] + h/2, y[i-1] + k1/2)
            k3 = h*self.f(x[i-1] + h, y[i-1] + k2)
            y[i] = y[i-1] + (k0 + 2*k1 + 2*k2 + k3)/6

        return zip(x, y)


    def rev_matrix(self, system) -> list:
        solution = np.dot(
            np.linalg.inv(
                np.delete(system, -1, axis=1)
            ),
            system[:, -1])

        return solution.tolist()


    def solve_edge_task(self, bounds, h, mfc, mfm, le, re):
        x = np.arange(bounds[0], bounds[1], h)
        equations, matrix = [None for i in range(len(x))], []
        variables = symbols(f'y:{len(x)}')

        equations[0] = Eq(
            le[0]*variables[0] + (le[1]/h)*(variables[1] - variables[0]),
            le[2]
        )

        equations[-1] = Eq(
            re[0] * variables[-1] + (re[1] / h) * (variables[-1] - variables[-2]),
            re[2]
        )

        for i in range(1, len(x) - 1):
            px_value, qx_value = mfm[0](x[i]), mfm[1](x[i])

            coefs = [mfc[0]/(h**2), mfc[1]*px_value/(2*h), mfc[2]*qx_value]

            equations[i] = Eq(
                coefs[0]*(variables[i+1] - 2*variables[i] + variables[i - 1]) + coefs[1]*(variables[i+1] - variables[i-1]) +coefs[2]*variables[i],
                mfc[-1]
            )

        for eq in equations:
            coefs = []
            coefs_dict = eq.lhs.as_coefficients_dict()
            for var in variables:
                coefs.append(coefs_dict.get(var, 0))
            coefs.append(eq.rhs)
            matrix.append(coefs)

        matrix = np.array(matrix).astype(np.float64)

        return self.rev_matrix(matrix), equations













