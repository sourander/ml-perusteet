import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import altair as alt
    import polars as pl
    import numpy as np

    from drawdata import ScatterWidget


@app.cell
def _():
    import marimo as mo

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    return LinearRegression, PolynomialFeatures, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solving Least Squares Problem

    This notebook lets you explore linear regression interactively. First, draw a dataset. Then, try to fit a line manually using sliders. Finally, compare your manual fit against analytical solutions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw the Dataset
    """)
    return


@app.cell
def _(mo):
    widget = mo.ui.anywidget(ScatterWidget())
    widget
    return (widget,)


@app.cell
def _(mo, widget):
    mo.stop(not widget.data, "You need to draw some data first. Use the drawdata widget above.")

    _raw = widget.data_as_polars
    _x_raw = _raw["x"].to_numpy()
    _y_raw = _raw["y"].to_numpy()

    # Normalize to range [0, 1]
    x_vals = (_x_raw - _x_raw.min()) / (_x_raw.max() - _x_raw.min())
    y_vals = (_y_raw - _y_raw.min()) / (_y_raw.max() - _y_raw.min())

    df = pl.DataFrame({"x": x_vals, "y": y_vals, "color": _raw["color"]})
    return df, x_vals, y_vals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual Fit

    Use the sliders below to set the slope (`m`) and intercept (`b`) of the line `y = mx + b`. Try to minimize the **SSE** (Sum of Squared Errors).
    """)
    return


@app.cell
def _(mo):
    m_slider = mo.ui.slider(
        start=-5, stop=5, step=0.1, value=0.5, label="m (slope)"
    )
    b_slider = mo.ui.slider(
        start=-2, stop=2, step=0.1, value=0.0, label="b (intercept)"
    )
    mo.hstack([m_slider, b_slider])
    return b_slider, m_slider


@app.cell(hide_code=True)
def _(b_slider, df, m_slider, x_vals, y_vals):
    m = m_slider.value
    b = b_slider.value

    y_pred_manual = m * x_vals + b
    sse_manual = float(np.sum((y_vals - y_pred_manual) ** 2))

    _scale_x = alt.Scale(domain=[-0.1, 1.1], nice=False)
    _scale_y = alt.Scale(domain=[-0.1, 1.1], nice=False)

    # Training data scatter
    scatter = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.85)
        .encode(
            x=alt.X("x:Q", scale=_scale_x),
            y=alt.Y("y:Q", scale=_scale_y),
            color=alt.Color("color:N", scale=None, legend=None),
        )
        .properties(width="container", height=400)
    )

    # Regression line
    line_x = np.linspace(0, 1, 200)
    line_y = m * line_x + b
    line_df = pl.DataFrame({"x": line_x, "y": line_y})

    line_chart = (
        alt.Chart(line_df)
        .mark_line(color="black", strokeWidth=2, clip=True)
        .encode(
            x=alt.X("x:Q", scale=_scale_x),
            y=alt.Y("y:Q", scale=_scale_y),
        )
    )

    # Residual lines (vertical dashed lines from each point to the line)
    residual_rows = []
    for i in range(len(x_vals)):
        residual_rows.append({"group": i, "x": float(x_vals[i]), "y": float(y_vals[i])})
        residual_rows.append({"group": i, "x": float(x_vals[i]), "y": float(y_pred_manual[i])})

    residual_df = pl.DataFrame(residual_rows)
    residual_chart = (
        alt.Chart(residual_df)
        .mark_line(color="red", strokeDash=[4, 2], opacity=0.6, clip=True)
        .encode(
            x=alt.X("x:Q", scale=_scale_x),
            y=alt.Y("y:Q", scale=_scale_y),
            detail="group:N",
        )
    )

    chart = scatter + residual_chart + line_chart
    chart
    return b, m, sse_manual


@app.cell(hide_code=True)
def _(b, m, mo, sse_manual):
    mo.callout(
        mo.md(f"**Equation:** `y = {m:.2f}x + {b:.2f}`\n\n**SSE:** {sse_manual:.2f}"),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analytical Solutions

    Compare three approaches for finding the best-fit line (or curve):

    1. **Normal Equation** — closed-form solution using matrix algebra
    2. **LinearRegression** — scikit-learn's implementation
    3. **Polynomial Regression** — `PolynomialFeatures` + `LinearRegression` (adjust degree with the slider)
    """)
    return


@app.cell
def _(mo):
    degree_slider = mo.ui.slider(start=1, stop=12, value=1, step=1, label="Polynomial Degree")
    degree_slider
    return (degree_slider,)


@app.cell(hide_code=True)
def _(LinearRegression, PolynomialFeatures, degree_slider, df, x_vals, y_vals):
    # --- 1. Normal Equation (linear: y = mx + b) ---
    X_mat = np.column_stack([x_vals, np.ones_like(x_vals)])
    coeffs = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vals
    m_ne, b_ne = coeffs[0], coeffs[1]

    line_xs = np.linspace(-0.1, 1.1, 200)
    y_ne = m_ne * line_xs + b_ne
    sse_ne = float(np.sum((y_vals - (m_ne * x_vals + b_ne)) ** 2))

    # --- 2. LinearRegression ---
    lr = LinearRegression()
    lr.fit(x_vals.reshape(-1, 1), y_vals)
    y_lr = lr.predict(line_xs.reshape(-1, 1))
    sse_lr = float(np.sum((y_vals - lr.predict(x_vals.reshape(-1, 1))) ** 2))

    # --- 3. PolynomialFeatures + LinearRegression ---
    degree = degree_slider.value
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly.fit_transform(x_vals.reshape(-1, 1))
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly_train, y_vals)

    X_poly_line = poly.transform(line_xs.reshape(-1, 1))
    y_poly = lr_poly.predict(X_poly_line)
    sse_poly = float(np.sum((y_vals - lr_poly.predict(X_poly_train)) ** 2))

    # --- Build chart ---
    _scale_x = alt.Scale(domain=[-0.1, 1.1])
    _scale_y = alt.Scale(domain=[-0.1, 1.1])

    _scatter = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.85)
        .encode(
            x=alt.X("x:Q", scale=_scale_x),
            y=alt.Y("y:Q", scale=_scale_y),
            color=alt.Color("color:N", scale=None, legend=None),
        )
        .properties(width="container", height=400)
    )

    solutions_df = pl.DataFrame({
        "x": np.tile(line_xs, 3),
        "y": np.concatenate([y_ne, y_lr, y_poly]),
        "method": (
            ["Normal Equation"] * len(line_xs)
            + ["LinearRegression"] * len(line_xs)
            + [f"Polynomial (degree={degree})"] * len(line_xs)
        ),
    })

    _solutions_chart = (
        alt.Chart(solutions_df)
        .mark_line(strokeWidth=2, clip=True)
        .encode(
            x=alt.X("x:Q", scale=_scale_x),
            y=alt.Y("y:Q", scale=_scale_y),
            color=alt.Color("method:N", legend=alt.Legend(title="Method")),
            strokeDash=alt.StrokeDash("method:N", legend=None),
        )
    )

    _chart = _scatter + _solutions_chart
    _chart
    return degree, sse_lr, sse_ne, sse_poly


@app.cell(hide_code=True)
def _(degree, mo, sse_lr, sse_ne, sse_poly):
    mo.callout(
        mo.md(
            f"| Method | SSE |\n"
            f"|--------|-----|\n"
            f"| Normal Equation | {sse_ne:.2f} |\n"
            f"| LinearRegression | {sse_lr:.2f} |\n"
            f"| Polynomial (degree={degree}) | {sse_poly:.2f} |"
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
