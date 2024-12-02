import numpy as np

class IntegralSolver:
    def __init__(self, func):
        self.func = func
        self.nodes = None
        self.n = None


    def make_splitting2(self, n: int, bounds: tuple):
        self.nodes = np.linspace(bounds[0], bounds[1], n)
        self.n = n


    def make_splitting3(self, n: int, bounds: tuple):
        self.nodes = np.linspace(bounds[0], bounds[1], 2*n)
        self.n = n


    def rectangle_method(self):
        return np.sum(
            [
                self.func(self.nodes[i])*(self.nodes[i+1] - self.nodes[i])
                for i in range(len(self.nodes)-1)
            ]
        )


    def trapezoid_method(self):
        return 0.5*np.sum(
            [
                (self.func(self.nodes[i]) + self.func(self.nodes[i+1]))*(self.nodes[i+1] - self.nodes[i])
                for i in range(self.n - 1)
            ]
        )



    def simpson_method(self):
        h = self.nodes[1] - self.nodes[0]
        return (h/3)*np.sum(
            [
                self.func(self.nodes[2*i]) + 4*self.func(self.nodes[2*i + 1]) + self.func(self.nodes[2*i + 2])
                for i in range(self.n - 1)
            ]
        )