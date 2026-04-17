import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import polars as pl
    import altair as alt

    from drawdata import ScatterWidget
    from dataclasses import dataclass

    @dataclass
    class Epoch:
        i: int
        MSE: float

    @dataclass
    class SGDResult:
        w: np.ndarray
        y_hat: np.ndarray
        epochs: list[Epoch]


@app.cell
def _():
    import marimo as mo

    from sklearn.preprocessing import StandardScaler

    return StandardScaler, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stochastic Gradient Descent from Scratch

    This notebook implements **Stochastic Gradient Descent (SGD)** for linear regression from scratch. Unlike Batch Gradient Descent, SGD updates weights using **one randomly selected sample** at a time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw the Dataset

    Use the widget below to draw points. The **x-axis** is the feature and the **y-axis** is the target. Draw a roughly linear pattern with ~30+ points.
    """)
    return


@app.cell
def _(mo):
    widget = mo.ui.anywidget(ScatterWidget(n_classes=1))
    widget
    return (widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore the Data
    """)
    return


@app.cell
def _(StandardScaler, mo, widget):
    mo.stop(
        not widget.data,
        mo.md("**Draw some points first** using the widget above."),
    )

    _raw = widget.data_as_polars

    # Extract raw values
    _x_raw = _raw["x"].to_numpy().reshape(-1, 1)
    _y_raw = _raw["y"].to_numpy().reshape(-1, 1)

    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_std = scaler_X.fit_transform(_x_raw)  # shape (n, 1)
    y_std = scaler_y.fit_transform(_y_raw).flatten()  # shape (n,)

    df = pl.DataFrame({
        "x (raw)": _x_raw.flatten(),
        "y (raw)": _y_raw.flatten(),
        "x (scaled)": X_std.flatten(),
        "y (scaled)": y_std,
    })

    df
    return X_std, y_std


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict with Dot Product

    For linear regression, the prediction is:

    $$\hat{y} = X\mathbf{w}$$

    We prepend a **bias column of ones** so that $w_0$ acts as the intercept.
    """)
    return


@app.function
def predict(X: np.ndarray, w: np.ndarray, add_bias: bool = True) -> np.ndarray:
    if add_bias:
        bias_column = np.ones((X.shape[0], 1))
        X = np.concatenate((bias_column, X), axis=1)
    return np.dot(X, w)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean Squared Error
    """)
    return


@app.cell
def _():
    def mse(residuals: np.ndarray) -> float:
        return float(np.mean(residuals ** 2))

    def mae(residuals: np.ndarray) -> float:
        return float(np.mean(np.abs(residuals)))

    return mae, mse


@app.cell
def _(X_std, mo, mse, y_std):
    # Random initial weights: [bias, w1]
    _rng = np.random.default_rng(42)
    initial_w = _rng.uniform(-1, 1, size=2)
    initial_y_hat = predict(X_std, initial_w)

    _initial_mse = mse(y_std - initial_y_hat)
    mo.callout(
        mo.md(f"**Initial MSE (random weights):** {_initial_mse:.2f}"),
        kind="warn",
    )
    return (initial_w,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Stochastic Gradient Descent

    The algorithm:

    1. Initialize weights randomly
    2. For each **epoch** (one full pass through the data):
        - **Shuffle** the training samples
        - For each sample $(x_i, y_i)$:
            - Compute prediction: $\hat{y}_i = \mathbf{x}_i \cdot \mathbf{w}$
            - Compute single-sample gradient: $\nabla = 2 \cdot \mathbf{x}_i \cdot (\hat{y}_i - y_i)$
            - Update weights: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \nabla$
    3. After each epoch, record the MSE over all data

    Unlike **Batch GD** (which uses all samples per update), SGD updates weights after every single sample. This makes it noisier but faster to converge.

    ### Gradient Clipping

    If the learning rate is too large, the weight updates overshoot and the weights grow exponentially — eventually reaching infinity. This is called **exploding gradients**. To prevent this, we use **gradient clipping**: if the norm of the gradient exceeds a threshold, we scale it down so that its norm equals the threshold. This limits the maximum step size without changing the gradient's direction.

    Use the sliders to change the number of epochs and the learning rate.
    """)
    return


@app.cell
def _(X_std, epochs_slider, initial_w, lr_radio, mse, y_std):
    def _sgd(X, y, n_epochs, learning_rate, clip_norm=5.0) -> SGDResult:
        rng = np.random.default_rng(42)
        w = initial_w.copy()
        n = len(y)
        epoch_history = []

        # Prepend bias column once
        _bias = np.ones((X.shape[0], 1))
        X_bias = np.concatenate((_bias, X), axis=1)

        for epoch in range(n_epochs):
            # Shuffle indices each epoch
            indices = rng.permutation(n)

            for i in indices:
                x_i = X_bias[i]       # shape (2,) — [1, x]
                y_i = y[i]            # scalar

                y_hat_i = x_i @ w     # scalar prediction
                gradient = 2 * x_i * (y_hat_i - y_i)  # shape (2,)

                # Gradient clipping: scale down if the gradient norm exceeds clip_norm.
                # This prevents exploding gradients when the learning rate is too high.
                grad_norm = np.linalg.norm(gradient)
                if grad_norm > clip_norm:
                    gradient = gradient * (clip_norm / grad_norm)

                w = w - learning_rate * gradient

            # Record MSE after full epoch
            _y_hat_all = X_bias @ w
            _epoch_mse = mse(y - _y_hat_all)
            epoch_history.append(Epoch(epoch, _epoch_mse))

        final_y_hat = predict(X, w)
        return SGDResult(w, final_y_hat, epoch_history)

    result = _sgd(X_std, y_std, epochs_slider.value, lr_radio.value)
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Progress

    MSE after each epoch. The noisy descent is characteristic of SGD.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    epochs_slider = mo.ui.slider(
        start=10,
        stop=500,
        step=10,
        value=100,
        label="Epochs",
    )
    _options = {"0.0001": 1e-4, "0.001": 1e-3, "0.01": 1e-2, "0.1": 0.1, "1.0": 1.0}
    lr_radio = mo.ui.radio(options=_options, value="0.0001", label="Learning rate (α)")
    mo.hstack([epochs_slider, lr_radio], justify="start", gap=2)
    return epochs_slider, lr_radio


@app.cell(hide_code=True)
def _(result):
    _progress_df = pl.DataFrame({
        "Epoch": [e.i for e in result.epochs],
        "MSE": [e.MSE for e in result.epochs],
    })

    # Reference lines for important MSE thresholds
    _ref_df = pl.DataFrame({
        "reference": [0.1, 1.0],
        "label": ["0.1", "1.0"],
    })

    line = alt.Chart(_progress_df).mark_line(
        strokeWidth=1.5,
        color="steelblue",
    ).encode(
        x=alt.X("Epoch:Q", title="Epoch"),
        y=alt.Y("MSE:Q", title="MSE"),
        tooltip=[
            alt.Tooltip("Epoch:Q"),
            alt.Tooltip("MSE:Q", format=".4f"),
        ],
    )

    rules = alt.Chart(_ref_df).mark_rule(
        color="firebrick",
        strokeDash=[6, 4],
        opacity=0.8,
    ).encode(
        y="reference:Q"
    )

    labels = alt.Chart(_ref_df).mark_text(
        align="left",
        dx=6,
        dy=-4,
        color="firebrick",
    ).encode(
        x=alt.value(6),   # pixel offset from left edge
        y="MSE:Q",
        text="label:N",
    )

    chart = (line + rules + labels).properties(
        width="container",
        height=350,
        title="MSE over Epochs",
    )

    chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Evaluation
    """)
    return


@app.cell
def _(mae, mo, mse, result, y_std):
    _final_mse = mse(y_std - result.y_hat)
    _final_rmse = float(np.sqrt(_final_mse))
    _final_mae = mae(y_std - result.y_hat)

    mo.callout(
        mo.md(
            f"**MSE:** {_final_mse:.4f}\n\n"
            f"**RMSE:** {_final_rmse:.4f}\n\n"
            f"**MAE:** {_final_mae:.4f}\n\n"
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predictions vs Actual

    This is a **general-purpose diagnostic chart** that works regardless of how many features the model has. The x-axis shows the true target value $y$ and the y-axis shows the model's prediction $\hat{y}$. If the model were perfect, every point would land on the red diagonal where $\hat{y} = y$. The orange dashed lines show the ±RMSE band.

    This same chart type appeared already in the Hill Climbing notebook (620), where the model had two features — making the fitted line chart below impossible.
    """)
    return


@app.cell(hide_code=True)
def _(mse, result, y_std):
    _final_mse = mse(y_std - result.y_hat)
    _rmse = float(np.sqrt(_final_mse))
    _max_val = float(max(y_std.max(), result.y_hat.max()))
    _min_val = float(min(y_std.min(), result.y_hat.min()))

    _scatter_df = pl.DataFrame({"y": y_std, "y_hat": result.y_hat})
    _scatter = (
        alt.Chart(_scatter_df)
        .mark_circle(size=20, opacity=0.5, color="steelblue")
        .encode(
            x=alt.X("y:Q", title="Actual (y)"),
            y=alt.Y("y_hat:Q", title="Predicted (ŷ)"),
            tooltip=[
                alt.Tooltip("y:Q", format=".2f"),
                alt.Tooltip("y_hat:Q", format=".2f"),
            ],
        )
    )

    _ideal_df = pl.DataFrame({"y": [_min_val, _max_val], "y_hat": [_min_val, _max_val]})
    _ideal_line = (
        alt.Chart(_ideal_df)
        .mark_line(color="red", strokeWidth=1.5)
        .encode(x="y:Q", y="y_hat:Q")
    )

    _upper_df = pl.DataFrame({
        "y": [_min_val, _max_val],
        "y_hat": [_min_val + _rmse, _max_val + _rmse],
    })
    _upper_line = (
        alt.Chart(_upper_df)
        .mark_line(color="orange", strokeDash=[6, 3], strokeWidth=1)
        .encode(x="y:Q", y="y_hat:Q")
    )

    _lower_df = pl.DataFrame({
        "y": [_min_val, _max_val],
        "y_hat": [_min_val - _rmse, _max_val - _rmse],
    })
    _lower_line = (
        alt.Chart(_lower_df)
        .mark_line(color="orange", strokeDash=[6, 3], strokeWidth=1)
        .encode(x="y:Q", y="y_hat:Q")
    )

    (_scatter + _ideal_line + _upper_line + _lower_line).properties(
        width="container", height=400, title="y vs ŷ"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitted Line

    This chart is only possible because our model has **a single feature**: we can plot $x$ on the horizontal axis and $y$ on the vertical axis, and draw the regression line directly over the data. With two or more features, this 2D visualization would no longer be possible.
    """)
    return


@app.cell(hide_code=True)
def _(X_std, result, y_std):
    _data_df = pl.DataFrame({
        "x": X_std.flatten(),
        "y": y_std,
    })

    _scatter = (
        alt.Chart(_data_df)
        .mark_circle(size=30, opacity=0.5, color="teal")
        .encode(
            x=alt.X("x:Q", title="x (scaled)"),
            y=alt.Y("y:Q", title="y (scaled)"),
        )
    )

    # Build the regression line from min to max x
    _x_range = np.linspace(X_std.min(), X_std.max(), 100).reshape(-1, 1)
    _y_line = predict(_x_range, result.w)

    _line_df = pl.DataFrame({
        "x": _x_range.flatten(),
        "y": _y_line,
    })

    _line = (
        alt.Chart(_line_df)
        .mark_line(color="red", strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    )

    (_scatter + _line).properties(
        width="container", height=400, title="Fitted Regression Line"
    )
    return


if __name__ == "__main__":
    app.run()
