import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt
    import numpy as np

    from math import pi


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cyclical Encoding

    In this Notebook, we will create a dataset that consist of 24 hours of data (`00:00`, `01:00`, ..., `23:00`). This hour-of-day will be encoded as explained in the lesson.

    ## Starting Point

    Notice that our starting point is a DataFrame consisting of 24 rows. Each row has the time values as you would typically find it in a CSV, Database or similar. The `hour_of_day` column contains only the extracted `hour` value of the original time field.

    This now has a serious problem: the hours `23` and `01` are `23 - 1 = 22` hours aparts from each other. This is obviously not true when looking at a real clock. The difference is only 2 hours (`23 -> 00 -> 01`)
    """)
    return


@app.cell
def _():
    df = pl.DataFrame(
        {
            "time_field": pl.time_range(
                start=pl.time(00),
                end=pl.time(23),
                interval="1h",
                eager=True,
            ),
        }
    )

    df = df.with_columns(
        hour_of_day = pl.col("time_field").dt.hour()
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Add sin and cos

    We would compute the values using Numpy like this...

    ```
    df["hour_sin"] = np.sin(df["hour_of_day"] / 24 * 2 * np.pi)
    df["hour_cos"] = np.cos(df["hour_of_day"] / 24 * 2 * np.pi)
    ```

    However, since our data is in a Polars DataFrame, we can do the same using Polars Expressions.
    """)
    return


@app.cell
def _(df):
    df_encoded = df.with_columns(
        hour_sin=pl.Expr.sin(pl.col("hour_of_day") / 24 * 2 * pi),
        hour_cos=pl.Expr.cos(pl.col("hour_of_day") / 24 * 2 * pi),
    )

    df_encoded
    return (df_encoded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot
    """)
    return


@app.cell(hide_code=True)
def _(df_encoded):
    chart = (
        alt.Chart(df_encoded)
        .transform_fold(
            ['hour_sin', 'hour_cos'],
            as_=['series', 'value']
        )
        .mark_point()
        .encode(
            x=alt.X('hour_of_day:Q', title='Hour of day'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('series:N', title='Series'),
            tooltip=[
                alt.Tooltip('hour_of_day:Q', format=',.0f', title='Hour'),
                alt.Tooltip('series:N', title='Series'),
                alt.Tooltip('value:Q', format=',.2f', title='Value')
            ]
        )
        .properties(
            height=290,
            width="container",
        )
    )

    chart

    return


@app.cell(hide_code=True)
def _(df_encoded):
    points = alt.Chart(df_encoded).mark_point(size=200, opacity=0.3).encode(
        x=alt.X('hour_sin:Q', scale=alt.Scale(domain=[-1.2, 1.2])),
        y=alt.Y('hour_cos:Q', scale=alt.Scale(domain=[-1.2, 1.2])),
    )

    labels = alt.Chart(df_encoded).mark_text(
        align='center',
        baseline='middle',
        fontSize=12
    ).encode(
        x='hour_sin:Q',
        y='hour_cos:Q',
        text=alt.Text('hour_of_day:Q', format='.0f'),
        tooltip=[
            alt.Tooltip('hour_of_day:Q'),
            alt.Tooltip('hour_sin:Q', format=".2f"),
            alt.Tooltip('hour_cos:Q', format=".2f"),
        ]
    )

    _chart = (points + labels).properties(height=350, width=350).configure_axis(grid=True)
    _chart
    return


if __name__ == "__main__":
    app.run()
