import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import altair as alt
    import polars as pl
    import numpy as np

    from dataclasses import dataclass

    @dataclass
    class Iteration:
        i: int
        MSE: float

    @dataclass
    class HillClimbResult:
        w: np.ndarray
        y_hat: np.ndarray
        iterations: list[Iteration]


@app.cell
def _():
    import marimo as mo

    from sklearn.datasets import make_regression
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    return MinMaxScaler, StandardScaler, make_regression, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hill Climb from Scratch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate the Dataset

    The dataset simulates snake bite incidents. Features are **snake length** (cm) and **temperature** (°C). The target is the number of **sick leave days**.
    """)
    return


@app.cell
def _(MinMaxScaler, make_regression):
    # Generate regression dataset (same as in the theory lesson)
    _X, _y = make_regression(n_samples=200, n_features=2, noise=10, random_state=666)

    # Move all values to positive
    _X = _X - _X.min()
    _y = _y - _y.min()

    # Round values
    _X = _X.round(2)
    _y = _y.round(0)

    # Scale first column (snake length) to 50–300 cm
    _scaler_cm = MinMaxScaler(feature_range=(50, 300))
    _X[:, 0] = _scaler_cm.fit_transform(_X[:, 0].reshape(-1, 1)).flatten()

    # Scale second column (temperature) to 0–35 °C
    _scaler_celsius = MinMaxScaler(feature_range=(0, 35))
    _X[:, 1] = _scaler_celsius.fit_transform(_X[:, 1].reshape(-1, 1)).flatten()

    X_raw = _X
    y = _y
    return X_raw, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore the Data
    """)
    return


@app.cell
def _(X_raw, y):
    df = pl.DataFrame({
        "snake_length": X_raw[:, 0],
        "temperature": X_raw[:, 1],
        "sick_days": y,
    })
    df.head(7)
    return (df,)


@app.cell(hide_code=True)
def _(df):
    # Correlation heatmap
    _corr = df.corr()
    _cols = df.columns

    _rows = []
    for _i, _row_name in enumerate(_cols):
        for _j, _col_name in enumerate(_cols):
            _rows.append({
                "Variable 1": _row_name,
                "Variable 2": _col_name,
                "Correlation": round(_corr[_i, _j], 2),
            })

    _corr_df = pl.DataFrame(_rows)

    _heatmap = (
        alt.Chart(_corr_df)
        .mark_rect()
        .encode(
            x=alt.X("Variable 1:N", title=None),
            y=alt.Y("Variable 2:N", title=None),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
            ),
            tooltip=[
                alt.Tooltip("Variable 1:N"),
                alt.Tooltip("Variable 2:N"),
                alt.Tooltip("Correlation:Q", format=".2f"),
            ],
        )
        .properties(width=300, height=300, title="Correlation Matrix")
    )

    _text = (
        alt.Chart(_corr_df)
        .mark_text(fontSize=14)
        .encode(
            x="Variable 1:N",
            y="Variable 2:N",
            text=alt.Text("Correlation:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Correlation > 0.5,
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    _heatmap + _text
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Standardize the Data

    Hill Climbing needs scaled features — large raw values would dominate the error and make the random weight perturbations ineffective. We use **z-score** (standard) scaling.
    """)
    return


@app.cell
def _(StandardScaler, X_raw):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    # Show a few rows of scaled data
    pl.DataFrame({
        "x[0] (length)": X_std[:, 0],
        "x[1] (temp)": X_std[:, 1],
    }).head(7)
    return (X_std,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict with Dot Product

    For multivariate linear regression, the prediction is:

    $$\hat{y} = Xw$$

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
    ### Initial Prediction

    Start with arbitrary weights (the same ones used in the theory lesson) and see how bad the first prediction is.
    """)
    return


@app.cell
def _(X_std, y):
    # Initial weights from the lesson
    initial_w = np.array([0.78824801, 0.01379396, 0.60234906])
    initial_y_hat = predict(X_std, initial_w)

    pl.DataFrame({
        "x[0]": X_std[:, 0],
        "x[1]": X_std[:, 1],
        "y": y,
        "y_hat": initial_y_hat,
    }).head(7)
    return initial_w, initial_y_hat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mean Squared Error
    """)
    return


@app.function
def mse(residuals: np.ndarray) -> float:
    return float(np.mean(residuals ** 2))


@app.cell
def _(initial_y_hat, mo, y):
    _initial_mse = mse(y - initial_y_hat)
    mo.callout(
        mo.md(f"**Initial MSE:** {_initial_mse:.2f}"),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hill Climb Algorithm

    The algorithm:

    1. Initialize weights (we use the same arbitrary weights as above)
    2. Compute the error (MSE)
    3. Add a random perturbation (uniform ±1.0) to all weights
    4. If the new MSE is lower, keep the new weights
    5. Repeat steps 3–4 for `max_iter` iterations

    Use the slider to change the number of iterations.
    """)
    return


@app.cell
def _(mo):
    max_iter_slider = mo.ui.slider(
        start=1_000,
        stop=50_000,
        step=1_000,
        value=10_000,
        label="max_iter",
    )
    max_iter_slider
    return (max_iter_slider,)


@app.cell
def _(X_std, initial_w, max_iter_slider, y):
    def hill_climb(X, y, max_iter) -> HillClimbResult:
        beneficial_iterations = []
        best_weights = initial_w.copy()
        best_predictions = predict(X, best_weights)
        best_loss = mse(y - best_predictions)

        for _i in range(max_iter):
            candidate_weights = (
                best_weights + np.random.uniform(-1.0, 1.0, best_weights.shape)
            )
            candidate_predictions = predict(X, candidate_weights)
            candidate_loss = mse(y - candidate_predictions)

            if candidate_loss < best_loss:
                best_weights = candidate_weights
                best_predictions = candidate_predictions
                best_loss = candidate_loss
                beneficial_iterations.append(Iteration(_i, candidate_loss))

        return HillClimbResult(best_weights, best_predictions, beneficial_iterations)

    result = hill_climb(X_std, y, max_iter_slider.value)
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Progress

    Each dot represents an iteration where the MSE improved.
    """)
    return


@app.cell(hide_code=True)
def _(result):
    _progress_df = pl.DataFrame({
        "Iteration": [it.i for it in result.iterations],
        "MSE": [it.MSE for it in result.iterations],
    })

    alt.Chart(_progress_df).mark_circle(size=8, opacity=0.6).encode(
        x=alt.X("Iteration:Q", title="Iteration"),
        y=alt.Y("MSE:Q", title="MSE"),
        tooltip=[
            alt.Tooltip("Iteration:Q"),
            alt.Tooltip("MSE:Q", format=".2f"),
        ],
    ).properties(width="container", height=350, title="MSE over Beneficial Iterations")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Evaluation
    """)
    return


@app.cell
def _(mo, result, y):
    _final_mse = mse(y - result.y_hat)
    _final_rmse = float(np.sqrt(_final_mse))

    mo.callout(
        mo.md(
            f"**MSE:** {_final_mse:.2f}\n\n"
            f"**RMSE:** {_final_rmse:.2f} days"
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predictions vs Actual

    The red line is the ideal (perfect prediction). Orange dashed lines show the ±RMSE band.
    """)
    return


@app.cell(hide_code=True)
def _(result, y):
    _final_mse = mse(y - result.y_hat)
    _rmse = float(np.sqrt(_final_mse))
    _max_val = float(max(y.max(), result.y_hat.max()))

    _scatter_df = pl.DataFrame({"y": y, "y_hat": result.y_hat})
    _scatter = (
        alt.Chart(_scatter_df)
        .mark_circle(size=20, opacity=0.5, color="steelblue")
        .encode(
            x=alt.X("y:Q", title="Actual (y)"),
            y=alt.Y("y_hat:Q", title="Predicted (ŷ)"),
            tooltip=[
                alt.Tooltip("y:Q", format=".1f"),
                alt.Tooltip("y_hat:Q", format=".1f"),
            ],
        )
    )

    _ideal_df = pl.DataFrame({"y": [0.0, _max_val], "y_hat": [0.0, _max_val]})
    _ideal_line = (
        alt.Chart(_ideal_df)
        .mark_line(color="red", strokeWidth=1.5)
        .encode(x="y:Q", y="y_hat:Q")
    )

    _upper_df = pl.DataFrame({
        "y": [0.0, _max_val],
        "y_hat": [_rmse, _max_val + _rmse],
    })
    _upper_line = (
        alt.Chart(_upper_df)
        .mark_line(color="orange", strokeDash=[6, 3], strokeWidth=1)
        .encode(x="y:Q", y="y_hat:Q")
    )

    _lower_df = pl.DataFrame({
        "y": [0.0, _max_val],
        "y_hat": [-_rmse, _max_val - _rmse],
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


if __name__ == "__main__":
    app.run()
