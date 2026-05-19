import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quick start (local)

    > Minimal example of MLForecast

    /// warning
    COPYRIGHT NOTE!


    This notebook is adapted from the Nixtla MLForecast tutorial: [Quick start (local)](https://github.com/Nixtla/statsforecast/blob/main/nbs/docs/getting-started/3_Getting_Started_complete_polars.ipynb). Reason for reshare is version locking and ease-of-use for students. The file is otherwise untouched, but has been converted to Marimo Notebook format and swapped Pandas to Polars.

    Original project: [mlforecast](https://github.com/Nixtla/mlforecast)

    Copyright © Nixtla

    Licensed under the Apache License, Version 2.0:
    https://www.apache.org/licenses/LICENSE-2.0
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Main concepts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The main component of mlforecast is the `MLForecast` class, which abstracts away:

    * Feature engineering and model training through `MLForecast.fit`
    * Feature updates and multi step ahead predictions through `MLForecast.predict`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data format
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data is expected to be a polars dataframe in long format, that is, each row represents an observation of a single series at a given time, with at least three columns:

    * `id_col`: column that identifies each series.
    * `target_col`: column that has the series values at each timestamp.
    * `time_col`: column that contains the time the series value was observed. These are usually timestamps, but can also be consecutive integers.

    Here we present an example using the classic Box & Jenkins airline data, which measures monthly totals of international airline passengers from 1949 to 1960 [1].
    """)
    return


@app.cell
def _():
    import polars as pl
    from utilsforecast.plotting import plot_series

    return pl, plot_series


@app.cell
def _(pl):
    df = pl.read_csv('https://datasets-nixtla.s3.amazonaws.com/air-passengers.csv', try_parse_dates=True)
    df.head()
    return (df,)


@app.cell
def _(df):
    df['unique_id'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here the `unique_id` column has the same value for all rows because this is a single time series, you can have multiple time series by stacking them together and having a column that differentiates them.

    We also have the `ds` column that contains the timestamps, in this case with a monthly frequency, and the `y` column that contains the series values in each timestamp.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Modeling
    """)
    return


@app.cell
def _(df, plot_series):
    fig = plot_series(df)
    fig
    return


@app.cell
def _():
    # fig.savefig('../../figs/quick_start_local__eda.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that the series has a clear trend, so we can take the first difference, i.e. take each value and subtract the value at the previous month. This can be achieved by passing an `mlforecast.target_transforms.Differences([1])` instance to `target_transforms`.

    We can then train a linear regression using the value from the same month at the previous year (lag 12) as a feature, this is done by passing `lags=[12]`.
    """)
    return


@app.cell
def _():
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    from sklearn.linear_model import LinearRegression

    return Differences, LinearRegression, MLForecast


@app.cell
def _(Differences, LinearRegression, MLForecast, df):
    fcst = MLForecast(
        models=LinearRegression(),
        freq='1mo',  # our series has a monthly frequency
        lags=[12],
        target_transforms=[Differences([1])],
    )
    fcst.fit(df)
    return (fcst,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The previous line computed the features and trained the model, so now we're ready to compute our forecasts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forecasting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Compute the forecast for the next 12 months
    """)
    return


@app.cell
def _(fcst):
    preds = fcst.predict(12)
    preds
    return (preds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize the results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can visualize what our prediction looks like.
    """)
    return


@app.cell
def _(df, plot_series, preds):
    fig_1 = plot_series(df, preds)
    fig_1
    return


@app.cell
def _():
    # fig_1.savefig('../../figs/quick_start_local__predictions.png', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And that's it! You've trained a linear regression to predict the air passengers for 1961.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    [1] Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series Analysis, Forecasting and Control. Third Edition. Holden-Day. Series G.
    """)
    return


if __name__ == "__main__":
    app.run()
