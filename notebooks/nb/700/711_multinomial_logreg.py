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
    # Multinomial Logistic Regression (Softmax Regression)

    Let's extend binary logistic regression to the multiclass case. We use a dataset with three
    features (body mass, height, age) and four individuals, classified into **three classes**.

    In the multiclass setting, the weight matrix $W$ has one **column per class**:

    $$Z = XW, \quad \hat{Y} = \text{softmax}(Z)$$

    where $X \in \mathbb{R}^{n \times p}$, $W \in \mathbb{R}^{p \times K}$, and
    $Z, \hat{Y} \in \mathbb{R}^{n \times K}$.

    For simplicity, we omit the bias term and work in the original feature scales.
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

    W = mo.ui.matrix(
        [
            [ 0.04, -0.01,  0.02],
            [-0.00,  0.02,  0.00],
            [-0.01,  0.01,  0.04],
        ],
        min_value=-1.0,
        max_value=1.0,
        precision=2,
        step=0.01,
        column_labels=["class_0", "class_1", "class_2"],
        label="$W$",
    )

    Y = mo.ui.matrix(
        [              # One Hot Encoded
            [0, 1, 0], # label 1
            [1, 0, 0], # label 0
            [0, 1, 0], # label 1
            [0, 0, 1], # label 2
        ],
        min_value=0,
        max_value=1,
        precision=0,
        step=1,
        column_labels=["class_0", "class_1", "class_2"],
        label="$Y$",
    )

    y = mo.ui.matrix(
        [
            [1],
            [0],
            [1],
            [2]
        ],
        min_value=0,
        max_value=2,
        precision=0,
        step=1,
        label="$y$"
    )

    mo.hstack([X, W, Y, y], justify="center", align="center", gap=3)
    return W, X, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward pass

    We compute the linear combination $Z = XW$ and then apply the **softmax** function
    row-wise to obtain class probabilities $\hat{Y}$.
    """)
    return


@app.function
# Important difference! Instead of sigmoid, we use softmax in multiclass regression!
def softmax(Z):
    # Numerically stable: subtract the row maximum before exponentiating
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


@app.cell
def _(W, X, Y, mo):
    X_np = np.array(X.value)
    W_np = np.array(W.value)
    Y_np = np.array(Y.value)

    # Bias trick: prepend a column of ones to X and a corresponding row of zeros to W
    X_np = np.hstack([np.ones((X_np.shape[0], 1)), X_np])
    W_np = np.vstack([np.zeros((1, W_np.shape[1])), W_np])

    Z_np = X_np @ W_np
    Y_hat = softmax(Z_np)

    _class_labels = ["class_0", "class_1", "class_2"]

    X_display = mo.ui.matrix(X_np, disabled=True, column_labels=["bias", "bmass", "height", "age"], label="$X$")
    W_display = mo.ui.matrix(W_np, disabled=True, column_labels=_class_labels, label="$W$", precision=3)
    Z_display = mo.ui.matrix(Z_np, disabled=True, column_labels=_class_labels, label="$Z$", precision=2)
    Y_hat_display = mo.ui.matrix(Y_hat, disabled=True, column_labels=_class_labels, label=r"$\hat{Y}_{proba}$", precision=3)

    mo.hstack(
        [X_display, W_display, Z_display, Y_hat_display],
        justify="center",
        align="center",
        gap=3,
    )
    return Y_hat, Y_np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ⚠️ Important difference to Binary Logistic Regression

    Notice that the $Y$ is a $3 \times 3$ matrix, and thus are also the raw logit values in the $\hat{Y}$. To get get actual prediction, we find the nth column that has the largest value.
    """)
    return


@app.cell
def _(Y_hat, mo):
    y_hat = Y_hat.argmax(axis=1)
    y_hat_display =  mo.ui.matrix(y_hat, disabled=True, label=r"$\hat{y}$", precision=0)
    y_hat_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compute loss

    The loss is **categorical cross-entropy**:

    $$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} Y_{ik} \log \hat{Y}_{ik}$$
    """)
    return


@app.cell
def _(Y_hat, Y_np):
    def categorical_cross_entropy(Y, Y_hat):
        eps = 1e-8  # avoid log(0)
        return -np.mean(np.sum(Y * np.log(Y_hat + eps), axis=1))

    loss = categorical_cross_entropy(Y_np, Y_hat)
    loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## YOUR TASK: Compute gradients

    You need to implement this as a student. Do not be lazy and ask this from an LLM.
    Make sure you understand what is happening. Do the following:

    * Derive and implement the gradient of the categorical cross-entropy loss with respect to $W$
    * Using a learning rate of 0.001, compute one gradient descent step
    * Display side-by-side using `mo.ui.matrix`:
      * The old weights (`W_display`)
      * The updated weights $W_{new}$
    * Calculate also the loss, and print:
      * The old loss (`loss`)
      * The new loss

    **Hint:** The gradient has the same elegant closed form as in the binary case — only the
    shapes change.
    """)
    return


@app.function
def softmax_cross_entropy_gradient(X, Y, Y_hat):
    pass


if __name__ == "__main__":
    app.run()
