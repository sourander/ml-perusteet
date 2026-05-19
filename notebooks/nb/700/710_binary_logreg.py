import marimo

__generated_with = "0.23.6"
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
    # Binary Logistic Regression

    Let's create a dataset which three columns (body mass, height, age) consisting of 4 individuals. For simplicity, we will not normalize the dataset, but we word on the original unit scales.

    Our `w` will be a column vector consisting of one weight per feature.
    """)
    return


@app.cell
def _(mo):
    X = mo.ui.matrix(
        [
            [70, 175, 25], 
            [80, 180, 30], 
            [60, 165, 22],
            [90, 185, 40],
        ],
        min_value=18,
        max_value=190,
        column_labels=["bmass", "height", "age"],
        step=1.0,
        precision=1,
        label="$X$",
    )

    w = mo.ui.matrix(
        [
            [-0.07],
            [0.03],
            [-0.01],
        ],
        min_value=-1.0,
        max_value=1.0,
        precision=2,
        step=0.01,
        label="$w$"
    )

    y = mo.ui.matrix(
        [
            [1],
            [1],
            [1],
            [0]
        ],
        min_value=0,
        max_value=1,
        precision=0,
        step=1,
        label="$y$"
    )

    mo.hstack([X, w, y], justify="center", align="center", gap=3)
    return X, w, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binary forward pass

    To compute the prediction, we will do the bias trick (add bias column and the corresponding weight) and then perform matmul using Numpy.
    """)
    return


@app.function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@app.cell
def _(X, mo, w, y):
    # To NumPy
    X_np = np.array(X.value)
    w_np = np.array(w.value)
    y_np = np.array(y.value)

    # Add bias to each
    # Make sure w is a column vector in NumPy too
    X_np = np.c_[np.ones((X_np.shape[0], 1)), X_np]
    w_np = np.insert(w_np, 0, 1.0).reshape(-1, 1)

    # Forward pass and logit
    z = X_np @ w_np
    y_hat_prob = sigmoid(z)

    # Display items
    X_np_display = mo.ui.matrix(X_np, disabled=True, column_labels=["bias", "bmass", "height", "age"], label="$X$")
    w_np_display = mo.ui.matrix(w_np, disabled=True, label="$w$", precision=3)
    y_hat_log_display = mo.ui.matrix(y_hat_prob, disabled=True, label="$\\hat{y}_{proba}$", precision=2)
    y_hat_bin_display = mo.ui.matrix((z > 0.5).astype(int), disabled=True, label="$\\hat{y}$")
    z_display = mo.ui.matrix(X_np @ w_np, disabled=True, label="$z$", precision=2)

    mo.hstack([
        X_np_display,
        w_np_display,
        z_display,
        y_hat_log_display,
        y_hat_bin_display
    ], justify="center", align="center", gap=3)
    return y_hat_prob, y_np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute loss

    Now we could compute the binary cross-entropy loss.
    """)
    return


@app.cell
def _(y_hat_prob, y_np):
    def binary_cross_entropy(y, y_hat):
        eps = 1e-8  # avoid log(0)
        loss = -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )
        return loss

    loss = binary_cross_entropy(y_np, y_hat_prob)
    loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## YOUR TASK: Compute gradients

    You need to implement this as a student. Do not be lazy and ask this from an LLM. Make sure you understand what is happening. Do the following:

    * Compute the gradients
    * Using a learning rate of 0.0001, compute the step
    * Display side-by-side using `mo.ui.matrix`:
      * The old weights (`w_np_display`)
      * The updated weights
    * Calculate also the loss, and print:
      * The old loss (`loss`)
      * The new loss
    """)
    return


@app.function
def binary_cross_entropy_gradient(X, y, y_hat):
    pass


if __name__ == "__main__":
    app.run()
