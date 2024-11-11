import math
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

class Interpolator:
    def __init__(self):
        self.points: list[tuple[float, float]] = []
        self.n = 0
        self.coefs = []
        self.biases = []
        self.conditions_set = []
        self.fspline = lambda x, conditions_set: [func(x, *c['coefs']) for c, cond, func in conditions_set if cond(x, *c['bounds'])][0]
        self.final_diffs = {}
        self.divided_diffs = {}
        self.table = PrettyTable()
        self.f = lambda x, coef, biases: coef * math.prod(
            [x - bias for bias in biases]
        )
        self.rounder = lambda x: round(x, 4)


    def load_points(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self.n = len(points)
        self.final_diffs[0] = [val[1] for val in self.points]
        self.divided_diffs[0] = [val[1] for val in self.points]


    def clear_memory(self):
        self.coefs.clear()
        self.biases.clear()
        self.conditions_set.clear()


    def build_lagrange(self):
        self.clear_memory()
        view = []
        for i, pnt in enumerate(self.points):
            upper = "*".join([f"(x - {self.points[j][0]})" for j in range(self.n) if i != j])
            self.biases.append(
                [self.points[j][0] for j in range(self.n) if i != j]
            )
            self.coefs.append(
                pnt[1] / math.prod([(pnt[0] - self.points[j][0]) if i != j else 1 for j in range(self.n)])
            )
            view.append(f"{self.rounder(self.coefs[i])}*{upper}")

        return f"L{self.n - 1}(x)=\n" + "\n+".join(view), lambda x: self.rounder(sum(
            [self.f(x, self.coefs[i], self.biases[i]) for i in range(self.n)])
        )


    def rebuild_table(self, name, field_names, rows) -> None:
        self.table.clear()
        self.table.title = name
        self.table.field_names = field_names
        self.table.add_rows(rows)


    def show_final_diffs(self, degree: int = 1) -> None:
        for k in range(1, degree + 1):
            self.final_diffs[k] = []
            for i in range(len(self.final_diffs[k - 1]) - 1):
                self.final_diffs[k].append(round(
                    self.final_diffs[k - 1][i+1] - self.final_diffs[k - 1][i],
                    4
                ))

        self.rebuild_table(
            "Таблица конечных разностей",
            list(self.final_diffs.keys()),
            np.array([diffs + [0]*(self.n - len(diffs)) for diffs in self.final_diffs.values()]).T
        )

        print(self.table)


    def show_divided_diffs(self, degree: int = 1) -> None:
        for k in range(1, degree + 1):
            self.divided_diffs[k] = []
            for i in range(len(self.divided_diffs[k - 1]) - 1):
                self.divided_diffs[k].append(round(
                    (self.divided_diffs[k - 1][i+1] - self.divided_diffs[k - 1][i])/(self.points[i+k][0] - self.points[i][0]),
                    4
                ))

        self.rebuild_table(
            "Таблица разделенных разностей",
            list(self.divided_diffs.keys()),
            np.array([diffs + [0] * (self.n - len(diffs)) for diffs in self.divided_diffs.values()]).T
        )

        print(self.table)


    def build_newton(self):
        self.clear_memory()
        view = []
        for diff in self.divided_diffs.items():
            view.append(
                f"{diff[1][0]}" + "*".join([f"(x - {self.points[j][0]})" for j in range(diff[0])])
            )
            self.coefs.append(diff[1][0])
            self.biases.append([self.points[j][0] for j in range(diff[0])])

        return f"N{self.n - 1}(x)=\n" + "\n+".join(view), lambda x: self.rounder(sum(
            [self.f(x, self.coefs[i], self.biases[i]) for i in range(self.n)])
        )


    def rev_matrix(self, system) -> list:
        solution = np.dot(
            np.linalg.inv(
                np.delete(system, -1, axis=1)
            ),
            system[:, -1])

        return solution.tolist()


    def build_linear_spline(self):
        self.clear_memory()
        view = ["F(x)="]
        for i in range(len(self.points) - 1):
            x0, x1, y0, y1 = self.points[i][0], self.points[i+1][0], self.points[i][1], self.points[i+1][1]
            a, b = map(self.rounder, self.rev_matrix(
                np.array([
                    [x0, 1, y0],
                    [x1, 1, y1]
                ]
                ).astype(np.float64)
            ))
            view.append(
                f"{a}x + {b}, {x0} <= x <= {x1}",
            )

            self.conditions_set.append([
                {'coefs': [a, b], 'bounds': [x0, x1]},
                lambda x, x0, x1: x0 <= x <= x1,
                lambda x, a, b: a*x + b
            ])

        return "\n".join(view), lambda x: self.fspline(x, self.conditions_set)

    def build_quadratic_spline(self):
        self.clear_memory()
        view = ["F(x)="]
        for i in range(0, len(self.points) - 1, 2):
            x0, x1, x2 = self.points[i][0], self.points[i + 1][0], self.points[i + 2][0]
            y0, y1, y2 = self.points[i][1], self.points[i + 1][1], self.points[i + 2][1]
            a, b, c = map(self.rounder, self.rev_matrix(
                np.array([
                    [x0**2, x0, 1, y0],
                    [x1**2, x1, 1, y1],
                    [x2 ** 2, x2, 1, y2]
                ]
                ).astype(np.float64)
            ))
            view.append(
                f"{a}x^2 + {b}x + {c}, {x0} <= x <= {x2}",
            )

            self.conditions_set.append([
                {'coefs': [a, b, c], 'bounds': [x0, x2]},
                lambda x, x0, x2: x0 <= x <= x2,
                lambda x, a, b, c: a*x**2 + b*x + c
            ])

        return "\n".join(view), lambda x: self.fspline(x, self.conditions_set)


    def show_graph(self, x, Y, names, styles) -> None:
        for y, name, style in zip(Y, names, styles.values()):
            plt.plot(x, y, color=style['color'], linestyle=style['linestyle'], label=name)

        plt.xlabel('oX')
        plt.ylabel('oY')
        plt.xticks(x.tolist())
        mn = min(self.points, key=lambda x: x[1])[1]
        mx = max(self.points, key=lambda x: x[1])[1]
        step = (mx - mn)/10
        plt.yticks(np.arange(mn, mx + step, step))
        plt.legend()
        plt.show()
