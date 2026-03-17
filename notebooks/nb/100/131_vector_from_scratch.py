import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Vector from Scratch

    This course will contain multiple Marimo Notebooks that implement something *from scratch*. The idea behind these is to enable students to look under-the-hood of various algorithms. This first one is about creating a Vector from scratch. We could also create e.g. Matrix and N-dimensional Tensor, if we would want, but Vector should be enough to get the idea.
    """)
    return


@app.class_definition
class Vector:
    def __init__(self, *args: int|float):
        self.elements = list(args)

    @staticmethod
    def add(a: "Vector", b: "Vector") -> "Vector":
        """
        Adds two vectors element-wise.
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")
        result = [x + y for x, y in zip(a, b)]
        return Vector(*result)

    @staticmethod
    def sub(a: "Vector", b: "Vector") -> "Vector":
        """
        Subtracts two vectors element-wise.
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")
        result = [x - y for x, y in zip(a, b)]
        return Vector(*result)

    @staticmethod
    def add_scalar(a, b):
        result = [x + b for x in a.elements]
        return Vector(*result)

    @staticmethod
    def sub_scalar(a, b):
        result = [x - b for x in a.elements]
        return Vector(*result)

    def concat(self, b: "Vector") -> "Vector":
        return Vector(*self.elements, *b.elements)

    def _check_len_match(self, other: "Vector"):
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length")

    def __add__(self, other):
        if isinstance(other, Vector):
            self._check_len_match(other)
            return self.add(self, other)
        # If int or float
        if isinstance(other, (int, float)):
            return self.add_scalar(self, other)
        else:
            raise NotImplementedError(f"Addition not supported for Vector and {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            self._check_len_match(other)
            return self.sub(self, other)
        if isinstance(other, (int, float)):
            return self.sub_scalar(self, other)
        else:
            raise NotImplementedError(f"Subtraction not supported for Vector and {type(other)}")

    def __rsub__(self, other):
        return self.__sub__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = [x / other for x in self.elements]
            return Vector(*result)
        else:
            raise NotImplementedError(f"Division not supported for Vector and {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = [x * other for x in self]
            return Vector(*result)
        if isinstance(other, Vector):
            self._check_len_match(other)
            result = [x * y for x, y in zip(self, other)]
            return Vector(*result)
        else:
            raise NotImplementedError(f"Multiplication not supported for Vector and {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        if isinstance(other, Vector):
            self._check_len_match(other)
            return sum(self * other)
        else:
            raise NotImplementedError(f"Matrix multiplication not supported for Vector and {type(other)}")

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            result = [x ** other for x in self]
            return Vector(*result)
        else:
            raise NotImplementedError(f"Power not supported for Vector and {type(other)}")

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return False
        return self.elements == other.elements

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def __setitem__(self, index, value):
        self.elements[index] = value

    def __repr__(self):
        return f"Vector({', '.join(map(str, self.elements))})"

    def __str__(self):
        return f"({', '.join(map(str, self.elements))})"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demonstation
    """)
    return


@app.cell
def _():
    v_1 = Vector(1, 2, 3)
    v_2 = Vector(4, 5, 6)

    print("\n===== Vector from Scratch ======")
    print("v_1 + v_2 = ", v_1 + v_2)
    print("v_1 +  42 = ", v_1 + 42)
    print("v_1 - v_2 = ", v_1 - v_2)
    print("v_1 /   2 = ", v_1 / 2)
    print("v_1 * v_2 = ", v_1 * v_2)
    print("v_1 @ v_2 = ", v_1 @ v_2)
    print("v_1 concat v_2 = ", v_1.concat(v_2))

    n_1 = np.array([1, 2, 3])
    n_2 = np.array([4, 5, 6])
    print("\n===== NumPy for Comparison ======")
    print("n_1 + n_2 = ", n_1 + n_2)
    print("n_1 +  42 = ", n_1 + 42)
    print("n_1 - n_2 = ", n_1 - n_2)
    print("n_1 /   2 = ", n_1 / 2)
    print("n_1 * n_2 = ", n_1 * n_2)
    print("n_1 @ n_2 = ", n_1 @ n_2)
    print("n_1 concat n_2 = ", np.concat((n_1, n_2)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix with Vector
    """)
    return


@app.cell
def _():
    def matrix_dot(X: list[Vector], w: Vector):
        return [x @ w for x in X]

    X = [
        Vector(11, 22),
        Vector(21, 22),
        Vector(31, 22),
    ]

    w = Vector(1, 2)

    y_hat = matrix_dot(X, w)
    print(y_hat)
    return


if __name__ == "__main__":
    app.run()
