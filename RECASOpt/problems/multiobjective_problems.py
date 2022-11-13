import math
import numpy as np
from abc import abstractmethod
from pySOT.optimization_problems import OptimizationProblem


class DTLZ(OptimizationProblem):
    def __init__(self, nobj = 2, dim = None):
        if dim is None: dim = nobj + 4
        self.dim = dim
        self.nobj = nobj

        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

        self.int_var = []
        self.cont_var = np.arange(0, dim)
        self.records = []

        self.minpt = None
        self.maxpt = None

    def transform(self, norm=False):
        self.minpt = np.asarray([min([record.fx[i] for record in self.records]) for i in range(self.nobj)])
        self.maxpt = np.asarray([max([record.fx[i] for record in self.records]) for i in range(self.nobj)])
        for record in self.records:
            record.bar_fx = record.fx - self.minpt if not norm else \
                (record.fx - self.minpt) / (self.maxpt - self.minpt + 1e-6)
            record.bar_fx += 1e-6

    @abstractmethod
    def eval(self, solution): pass


class DTLZ1(DTLZ):
    def __init__(self, nobj = 2, dim = None):
        super().__init__(nobj=nobj, dim=dim)
        self.name = "DTLZ1"

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum(
            [math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim - k:]]))
        f = [0.5 * (1.0 + g)] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= solution[j]
            if i > 0:
                f[i] *= 1 - solution[self.nobj-i-1]
        f = np.asarray(f)
        return f


class DTLZ2(DTLZ):
    def __init__(self, nobj = 2, dim = None):
        super().__init__(nobj=nobj, dim=dim)
        self.name = "DTLZ2"

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution[self.nobj - i - 1])
        f = np.asarray(f)
        return f


class DTLZ3(DTLZ):
    def __init__(self, nobj = 2, dim = None):
        super().__init__(nobj=nobj, dim=dim)
        self.name = "DTLZ3"

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 100.0 * (k + sum(
            [math.pow(x - 0.5, 2.0) - math.cos(20.0 * math.pi * (x - 0.5)) for x in solution[self.dim - k:]]))
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * solution[j])
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * solution[self.nobj - i - 1])
        f = np.asarray(f)
        return f


class DTLZ4(DTLZ):
    def __init__(self, nobj = 2, dim = None):
        super().__init__(nobj=nobj, dim=dim)
        self.name = "DTLZ4"

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        alpha = 100.0
        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi * math.pow(solution[j], alpha))
            if i > 0:
                f[i] *= math.sin(0.5 * math.pi * math.pow(solution[self.nobj - i - 1], alpha))
        f = np.asarray(f)
        return f


class DTLZ5(DTLZ):
    def __init__(self, nobj = 2, dim = None):
        super().__init__(nobj=nobj, dim=dim)
        self.name = "DTLZ5"

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x - 0.5, 2.0) for x in solution[self.dim-k:]])
        f = [1.0 + g]*self.nobj

        for i in range(self.nobj):
            for j in range(1, self.nobj-i-1):
                f[i] *= math.cos(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[j]))

            if i > 0:
                if self.nobj-i-1 != 0:
                    f[i] *= math.sin(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[self.nobj-i-1]))
                else:
                    f[i] *= math.sin(0.5 * math.pi * solution[0])

            if self.nobj - i - 1 != 0:
                f[i] *= math.cos(0.5 * math.pi * solution[0])

        f = np.asarray(f)
        return f


class DTLZ6(DTLZ):
    def __init__(self, nobj = 2, dim = None):
        super().__init__(nobj=nobj, dim=dim)
        self.name = "DTLZ6"

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = sum([math.pow(x, 0.1) for x in solution[self.dim - k:]])
        f = [1.0 + g] * self.nobj

        for i in range(self.nobj):
            for j in range(1, self.nobj - i - 1):
                f[i] *= math.cos(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[j]))

            if i > 0:
                if self.nobj - i - 1 != 0:
                    f[i] *= math.sin(0.5 * math.pi / (2.0 * (1.0 + g)) * (1.0 + 2.0 * g * solution[self.nobj - i - 1]))
                else:
                    f[i] *= math.sin(0.5 * math.pi * solution[0])

            if self.nobj - i - 1 != 0:
                f[i] *= math.cos(0.5 * math.pi * solution[0])

        f = np.asarray(f)
        return f


class DTLZ7(DTLZ):
    def __init__(self, nobj = 2, dim = None):
        super().__init__(nobj=nobj, dim=dim)
        self.name = "DTLZ7"

    def eval(self, solution):
        if len(solution) != self.dim:
            raise ValueError('Dimension mismatch')
        k = self.dim - self.nobj + 1
        solution = list(solution)

        g = 1.0 + (sum([x for x in solution[self.dim-k:]])) * 9.0 / float(k)
        f = [1.0] * self.nobj

        for i in range(self.nobj):
            if i < self.nobj - 1:
                f[i] = solution[i]
            else:
                h = 0
                for j in range(self.nobj - 1):
                    h += f[j] / (1.0 + g) * (1.0 + np.sin(3.0 * np.pi * f[j]))
                h = self.nobj - h
                f[i] = (1.0 + g) * h

        f = np.asarray(f)
        return f


if __name__ == '__main__':
    pass
