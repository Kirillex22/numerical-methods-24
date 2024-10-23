from prettytable import prettytable, PrettyTable

class Solver:

    def __init__(self):
        self.target_function = None
        self.derivatives = None
        self.eps: float = 0.01
        self.log = {}
        self.degree = 3
        self.rounder = lambda x: round(x, self.degree)


    def get_log(self, method_name: str):
        log: list = self.log.pop(method_name)
        table = PrettyTable()
        table.field_names = log[0]
        table.add_rows(log[1:])
        return table


    def information_collector(self, method_name: str, headers: list, iter_values: list):
        if not self.log.get(method_name, None):
            self.log[method_name] = [headers]

        self.log[method_name].append(list(map(self.rounder, iter_values)))


    def dividing_in_half(self, a: float, b: float) -> float:
        while abs(b-a) > self.eps:
            _a, _b = a, b
            x_center = (a+b)/2
            center: float = self.target_function(x_center)
            right: float = self.target_function(b)
            if center*right < 0:
                a = x_center
            else:
                b = x_center

            self.information_collector(
                'dividing_in_half',
                ['a', 'b', 'x_0', 'F(x_0)', '|b - a|', 'eps'],
                [_a, _b, x_center, center, abs(b-a), self.eps]
            )

        return self.rounder((a+b)/2)


    def newton(self, a: float, b: float) -> float:
        current_x = a if self.target_function(a)*self.derivatives[1](a) > 0 else b

        while abs(self.target_function(current_x)) > self.eps:
            current_x -= self.target_function(current_x)/self.derivatives[0](current_x)

            self.information_collector(
                'newton',
                ['x', 'F(x)', "F'(x)", 'eps'],
                [current_x, self.target_function(current_x), self.derivatives[0](current_x), self.eps]
            )

        return self.rounder(current_x)


    def simple_iteration(self, current_x: float) -> float:
        m = 1.01 * self.derivatives[0](current_x)
        f = lambda x: x - self.target_function(x)/m
        previous_x = current_x + self.eps + 1

        while abs(current_x - previous_x) > self.eps:
            previous_x = current_x
            current_x = f(current_x)
            self.information_collector(
                'simple_iteration',
                ['x_i-1', 'x_i', 'm', 'eps'],
                [previous_x, current_x, m, self.eps]
            )

        return self.rounder(current_x)


    def stephensen(self, current_x: float) -> float:
        previous_x = current_x + self.eps + 1
        while abs(current_x - previous_x) > self.eps:
            previous_x = current_x
            f_current_x = self.target_function(current_x)
            current_x -= f_current_x**2/(self.target_function(current_x+f_current_x) - f_current_x)
            self.information_collector(
                'stephensen',
                ['x', 'eps'],
                [current_x, self.eps]
            )

        return self.rounder(current_x)


    def modified_newton(self, current_x: float, h: float) -> float:
        while abs(self.target_function(current_x)) > self.eps:
            current_x -= self.target_function(current_x)*h/(self.target_function(current_x+h)-self.target_function(current_x))
            self.information_collector(
                'modified_newton',
                ['x', 'eps'],
                [current_x, self.eps]
            )

        return self.rounder(current_x)