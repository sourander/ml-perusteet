import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import polars as pl
    import altair as alt
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic regression – interactive visualisation

    In this notebook, you can explore how weights $w_1$, $w_2$, and bias $b$ affect
    the logistic regression **decision boundary** and **cross-entropy loss**. The data consists of
    the first 100 samples from the iris dataset – iris setosa and iris versicolor – on two features:
    sepal length ($x_1$) and width ($x_2$), both z-scored.

    The model is of the form:

    $$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

    Adjust the sliders and watch in real-time how the decision boundary and loss change.
    """)
    return


@app.cell
def _():
    _iris = datasets.load_iris()
    _X_raw = _iris.data[:100, :2]
    _y_raw = _iris.target[:100]

    _ss = StandardScaler()
    X_scaled = _ss.fit_transform(_X_raw)

    # Bias column appended at the end – same convention as in the lesson material.
    # Weight vector convention: w = [w_sepal_length, w_sepal_width, bias]
    X = np.c_[X_scaled, np.ones(X_scaled.shape[0])]
    y = _y_raw.astype(float)
    return X, X_scaled, y


@app.function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@app.function
def binary_cross_entropy(y_hat, y, eps=1e-15):
    y_clip = np.clip(y_hat, eps, 1 - eps)
    m = len(y)
    return float(
        -(1 / m) * np.sum(y * np.log(y_clip) + (1 - y) * np.log(1 - y_clip))
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adjust weights and bias

    The default values (`w1 = 5.67`, `w2 = 2.34`, `b = 0.67`) correspond to the intentionally
    poor starting point before gradient descent in the course material – the same situation shown
    in Figure 4 of the material. Try to find a better boundary by adjusting the sliders!

    The **threshold** does not affect cross-entropy loss, but determines which
    probabilities are classified as class 1 (iris versicolor).
    """)
    return


@app.cell
def _(mo):
    w1_slider = mo.ui.slider(
        start=-10.0,
        stop=10.0,
        step=0.1,
        value=5.67,
        label="$w_1$ (sepal length)",
    )
    w2_slider = mo.ui.slider(
        start=-10.0,
        stop=10.0,
        step=0.1,
        value=2.34,
        label="$w_2$ (sepal width)",
    )
    bias_slider = mo.ui.slider(
        start=-10.0,
        stop=10.0,
        step=0.1,
        value=0.67,
        label="$b$ (bias)",
    )
    threshold_slider = mo.ui.slider(
        start=0.02,
        stop=0.98,
        step=0.01,
        value=0.5,
        label="Threshold",
    )

    mo.vstack([
        mo.hstack([w1_slider, w2_slider], justify="start", gap=4),
        mo.hstack([bias_slider, threshold_slider], justify="start", gap=4),
    ])
    return bias_slider, threshold_slider, w1_slider, w2_slider


@app.cell
def _(X, bias_slider, threshold_slider, w1_slider, w2_slider, y):
    w = np.array([w1_slider.value, w2_slider.value, bias_slider.value])
    _threshold = threshold_slider.value

    z = X @ w
    y_hat = sigmoid(z)
    y_pred = (y_hat >= _threshold).astype(int)

    loss = binary_cross_entropy(y_hat, y)
    accuracy = float(np.mean(y_pred == y))
    return accuracy, loss, w, y_hat, y_pred, z


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatter plot and decision boundary
    """)
    return


@app.cell
def _(X_scaled, accuracy, loss, threshold_slider, w, y, y_hat, y_pred):
    _species = {0: "iris setosa", 1: "iris versicolor"}
    _correctness = ["correct" if p == int(t) else "incorrect" for p, t in zip(y_pred, y)]

    _scatter_df = pl.DataFrame({
        "sepal_length_std": X_scaled[:, 0].tolist(),
        "sepal_width_std": X_scaled[:, 1].tolist(),
        "species": [_species[int(yi)] for yi in y],
        "prediction": _correctness,
        "probability": [round(float(p), 4) for p in y_hat],
    })

    # Decision boundary: w1*x1 + w2*x2 + b = 0  →  x2 = -(w1*x1 + b) / w2
    _w1, _w2, _b = float(w[0]), float(w[1]), float(w[2])
    _threshold = threshold_slider.value

    _x1_min, _x1_max = -3.0, 3.0
    _x2_min, _x2_max = -3.0, 3.0

    _x1_line = np.linspace(_x1_min, _x1_max, 300)

    _x2_line = -(_w1 * _x1_line + _b) / _w2
    _boundary_df = pl.DataFrame({
        "sepal_length_std": _x1_line.tolist(),
        "sepal_width_std": _x2_line.tolist(),
    })

    # Threshold boundary: sigmoid(z) = t  →  z = logit(t)
    if abs(_w2) > 1e-6 and 0 < _threshold < 1:
        _logit_t = float(np.log(_threshold / (1.0 - _threshold)))
        _x2_thresh = -(_w1 * _x1_line + _b - _logit_t) / _w2
        _threshold_boundary_df = pl.DataFrame({
            "sepal_length_std": _x1_line.tolist(),
            "sepal_width_std": _x2_thresh.tolist(),
        })
    else:
        _threshold_boundary_df = pl.DataFrame({
            "sepal_length_std": [float("nan")],
            "sepal_width_std": [float("nan")],
        })

    _scatter = (
        alt.Chart(_scatter_df)
        .mark_point(size=80, strokeWidth=1.5, filled=True, opacity=0.85)
        .encode(
            x=alt.X(
                "sepal_length_std:Q",
                title="Sepal length (standardised)",
                scale=alt.Scale(domain=[_x1_min, _x1_max]),
            ),
            y=alt.Y(
                "sepal_width_std:Q",
                title="Sepal width (standardised)",
                scale=alt.Scale(domain=[_x2_min, _x2_max]),
            ),
            color=alt.Color(
                "species:N",
                title="Species",
                scale=alt.Scale(
                    domain=["iris setosa", "iris versicolor"],
                    range=["steelblue", "darkorange"],
                ),
            ),
            shape=alt.Shape(
                "prediction:N",
                title="Prediction",
                scale=alt.Scale(
                    domain=["correct", "incorrect"],
                    range=["circle", "cross"],
                ),
            ),
            tooltip=[
                alt.Tooltip("species:N", title="Species"),
                alt.Tooltip("prediction:N", title="Prediction"),
                alt.Tooltip("probability:Q", title="P(versicolor)", format=".3f"),
                alt.Tooltip("sepal_length_std:Q", title="x₁ (std)", format=".2f"),
                alt.Tooltip("sepal_width_std:Q", title="x₂ (std)", format=".2f"),
            ],
        )
    )

    _boundary_line = (
        alt.Chart(_boundary_df)
        .mark_line(color="firebrick", strokeWidth=2.0, strokeDash=[6, 4], clip=True)
        .encode(
            x=alt.X("sepal_length_std:Q"),
            y=alt.Y("sepal_width_std:Q"),
        )
    )

    _threshold_line = (
        alt.Chart(_threshold_boundary_df)
        .mark_line(color="purple", strokeWidth=2.0, opacity=0.5, clip=True)
        .encode(
            x=alt.X("sepal_length_std:Q"),
            y=alt.Y("sepal_width_std:Q"),
        )
    )

    _chart = (
        alt.layer(_scatter, _boundary_line, _threshold_line)
        .properties(
            width="container",
            height=420,
            title=f"Iris dataset – loss: {loss:.4f} – acc: {accuracy:.2f}",
        )
    )

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sigmoid panel: z → probability
    """)
    return


@app.cell
def _(accuracy, loss, threshold_slider, y, y_hat, y_pred, z):
    _species = {0: "iris setosa", 1: "iris versicolor"}
    _correctness = ["correct" if p == int(t) else "incorrect" for p, t in zip(y_pred, y)]
    _threshold = threshold_slider.value

    _sigmoid_df = pl.DataFrame({
        "z": z.tolist(),
        "probability": [round(float(p), 4) for p in y_hat],
        "species": [_species[int(yi)] for yi in y],
        "prediction": _correctness,
    })

    # Smooth sigmoid curve
    _z_pad = 1.0
    _z_range = np.linspace(float(z.min()) - _z_pad, float(z.max()) + _z_pad, 300)
    _sigmoid_curve_df = pl.DataFrame({
        "z": _z_range.tolist(),
        "probability": sigmoid(_z_range).tolist(),
    })

    # Reference: vertical line at z = 0
    _vline_zero_df = pl.DataFrame({"z": [0.0, 0.0], "probability": [0.0, 1.0]})

    # Threshold visuals
    _logit_t = float(np.log(_threshold / (1.0 - _threshold))) if 0 < _threshold < 1 else 0.0
    _hline_thresh_df = pl.DataFrame({
        "z": [float(_z_range[0]), float(_z_range[-1])],
        "probability": [_threshold, _threshold],
    })
    _vline_thresh_df = pl.DataFrame({"z": [_logit_t, _logit_t], "probability": [0.0, 1.0]})

    _z_domain = [float(_z_range[0]), float(_z_range[-1])]

    _curve = (
        alt.Chart(_sigmoid_curve_df)
        .mark_line(color="gray", strokeWidth=2.0)
        .encode(
            x=alt.X("z:Q", title="z = w₁x₁ + w₂x₂ + b", scale=alt.Scale(domain=_z_domain)),
            y=alt.Y("probability:Q", title="σ(z) = P(versicolor)", scale=alt.Scale(domain=[0, 1])),
        )
    )

    _points = (
        alt.Chart(_sigmoid_df)
        .mark_point(size=80, strokeWidth=1.5, filled=True, opacity=0.85)
        .encode(
            x=alt.X("z:Q", scale=alt.Scale(domain=_z_domain)),
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "species:N",
                title="Species",
                scale=alt.Scale(
                    domain=["iris setosa", "iris versicolor"],
                    range=["steelblue", "darkorange"],
                ),
            ),
            shape=alt.Shape(
                "prediction:N",
                title="Prediction",
                scale=alt.Scale(
                    domain=["correct", "incorrect"],
                    range=["circle", "cross"],
                ),
            ),
            tooltip=[
                alt.Tooltip("species:N", title="Species"),
                alt.Tooltip("prediction:N", title="Prediction"),
                alt.Tooltip("z:Q", title="z", format=".3f"),
                alt.Tooltip("probability:Q", title="P(versicolor)", format=".3f"),
            ],
        )
    )

    _vline_zero = (
        alt.Chart(_vline_zero_df)
        .mark_line(color="firebrick", strokeWidth=1.5, strokeDash=[6, 4])
        .encode(
            x=alt.X("z:Q"),
            y=alt.Y("probability:Q"),
        )
    )

    _hline_thresh = (
        alt.Chart(_hline_thresh_df)
        .mark_line(color="purple", strokeWidth=1.5, opacity=0.6, strokeDash=[4, 3])
        .encode(
            x=alt.X("z:Q"),
            y=alt.Y("probability:Q"),
        )
    )

    _vline_thresh = (
        alt.Chart(_vline_thresh_df)
        .mark_line(color="purple", strokeWidth=1.5, opacity=0.6)
        .encode(
            x=alt.X("z:Q"),
            y=alt.Y("probability:Q"),
        )
    )

    _sigmoid_chart = (
        alt.layer(_curve, _vline_zero, _hline_thresh, _vline_thresh, _points)
        .properties(
            width="container",
            height=300,
            title=f"Sigmoid panel \u2013 loss: {loss:.4f} \u2013 acc: {accuracy:.2f}",
        )
    )

    _sigmoid_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to read diagrams:

    ### Scatter Plot

    The dashed red line is the **decision boundary**: where $w_1 x_1 + w_2 x_2 + b = 0$,
    the sigmoid function equals exactly $0.5$. When the threshold is $0.5$, this line
    divides the classified region into two parts.

    - **Color** = true species
    - **Shape**: ○ = prediction correct, ✕ = prediction incorrect
    - **Tooltip**: hover over a point to see the predicted probability $\hat{y}$

    The **purple solid line** shows the current threshold boundary, where $\hat{y} = t$,
    i.e. $w_1 x_1 + w_2 x_2 + b = \text{logit}(t)$. When the threshold is $0.5$, the two
    lines coincide. Moving the threshold slider shifts the purple line relative to the red one.

    ### Sigmoid Curve

    Each sample is projected onto the sigmoid curve. The x-axis shows the linear score
    $z = w_1 x_1 + w_2 x_2 + b$ and the y-axis shows $\hat{y} = \sigma(z)$.

    - **Dashed red vertical line** at $z = 0$: where $\sigma(z) = 0.5$
    - **Purple horizontal line** at $\hat{y} = t$: current threshold
    - **Purple vertical line** at $z = \text{logit}(t)$: where samples are classified as class 1

    Hover over a point to see its score and predicted probability.
    """)
    return


if __name__ == "__main__":
    app.run()
