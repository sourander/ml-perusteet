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
    # Numpy & Pandas Basics

    Note that you are expected to have experience with Numpy, since that is part of KAMK's curriculum. This Marimo Notebook will function as a recap. This will also introduce the lesson's theory bits in practice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix without libraries

    Let's perform a simple linear algebra operation using Python-only solution. The operation we will perform is a dot product, like..

    $$
    \hat{y} = Xw
    $$

    Where the matrix X is...

    $$
    \begin{bmatrix}
    \hat{y}_{1} \\
    \hat{y}_{2} \\
    \hat{y}_{3} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    x_{1,1} & x_{1,2} \\
    x_{2,1} & x_{2,2} \\
    x_{3,1} & x_{3,2} \\
    \end{bmatrix}
    \begin{bmatrix}
    w_{1} \\
    w_{2} \\
    \end{bmatrix}
    $$

    And the operation is performed as...


    $$
    \begin{bmatrix}
    \hat{y}_{1} \\
    \hat{y}_{2} \\
    \hat{y}_{3} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    11 & 22 \\
    21 & 22 \\
    31 & 22 \\
    \end{bmatrix}
    \begin{bmatrix}
    1 \\
    2 \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    11 \cdot 1 + 22 \cdot 2 \\
    21 \cdot 1 + 22 \cdot 2 \\
    31 \cdot 1 + 22 \cdot 2 \\
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _():
    def vector_dot(u, v):
        return sum(u_i * v_i for u_i, v_i in zip(u, v))

    def matrix_dot(X, w):
        return [vector_dot(x, w) for x in X]

    # Matrix representation of a dataset with 3 rows, 2 features
    X_py = [
        (11, 22),
        (21, 22),
        (31, 22),
    ]

    # Vector (row-direction) vector representation of some parameters/coefficients
    w_py = (1, 2)

    y_hat_py = matrix_dot(X_py, w_py)
    print(y_hat_py)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix with NumPy
    """)
    return


@app.cell
def _():
    X = np.array(
    [
        (11, 22),
        (21, 22),
        (31, 22),
    ]
    )

    w = np.array([1, 2])

    # Option 1: Using the dot method
    y_hat = X.dot(w)

    # Option 2: Using the matmul operator
    y_hat = X @ w
    print(y_hat)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
