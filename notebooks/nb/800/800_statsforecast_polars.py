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
    # End to End Walkthrough with Polars

    > Model training, evaluation and selection for multiple time series

    /// warning
    COPYRIGHT NOTE!


    This notebook is adapted from the Nixtla StatsForecast tutorial: [End to End Walkthrough with Polars](https://github.com/Nixtla/statsforecast/blob/main/nbs/docs/getting-started/3_Getting_Started_complete_polars.ipynb). Reason for reshare is version locking and ease-of-use for students. The file is otherwise untouched, but has been converted to Marimo Notebook format.

    Original project: [StatsForecast](https://github.com/Nixtla/statsforecast)

    Copyright © Nixtla

    Licensed under the Apache License, Version 2.0:
    https://www.apache.org/licenses/LICENSE-2.0
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introducing Polars: A High-Performance DataFrame Library

    This document aims to highlight the recent integration of Polars, a robust and high-speed DataFrame library developed in Rust, into the functionality of StatsForecast. Polars, with its nimble and potent capabilities, has rapidly established a strong reputation within the Data Science community, further solidifying its position as a reliable tool for managing and manipulating substantial data sets.

    Available in languages including Rust, Python, Node.js, and R, Polars demonstrates a remarkable ability to handle sizable data sets with efficiency and speed that surpasses many other DataFrame libraries, such as Pandas. Polars' open-source nature invites ongoing enhancements and contributions, augmenting its appeal within the data science arena.

    The most significant features of Polars that contribute to its rapid adoption are:

    1. **Performance Efficiency**: Constructed using Rust, Polars exhibits an exemplary ability to manage substantial datasets with remarkable speed and minimal memory usage.

    2. **Lazy Evaluation**: Polars operates on the principle of 'lazy evaluation', creating an optimized logical plan of operations for efficient execution, a feature that mirrors the functionality of Apache Spark.

    3. **Parallel Execution**: Demonstrating the capability to exploit multi-core CPUs, Polars facilitates parallel execution of operations, substantially accelerating data processing tasks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisites
    This Guide assumes basic familiarity with StatsForecast. For a minimal example visit the [Quick Start](./getting_started_short.html)

    Follow this article for a step-by-step guide on building a production-ready forecasting pipeline for multiple time series.

    During this guide you will gain familiarity with the core `StatsForecast`class and some relevant methods like `StatsForecast.plot`, `StatsForecast.forecast` and `StatsForecast.cross_validation.`

    We will use a classical benchmarking dataset from the M4 competition. The dataset includes time series from different domains like finance, economy and sales. In this example, we will use a subset of the Hourly dataset.

    We will model each time series individually. Forecasting at this level is also known as local forecasting. Therefore, you will train a series of models for every unique series and then select the best one. StatsForecast focuses on speed, simplicity, and scalability, which makes it ideal for this task.

    **Outline:**

    1. Install packages.
    1. Read the data.
    2. Explore the data.
    3. Train many models for every unique combination of time series.
    4. Evaluate the model's performance using cross-validation.
    5. Select the best model for every unique time series.

    ## Not Covered in this guide

    * Forecasting at scale using clusters on the cloud.
        * [Forecast the M5 Dataset in 5min](../experiments/ets_ray_m5.html) using Ray clusters.
        * [Forecast the M5 Dataset in 5min](../experiments/prophet_spark_m5.html) using Spark clusters.
        * Learn how to predict [1M series in less than 30min](https://www.anyscale.com/blog/how-nixtla-uses-ray-to-accurately-predict-more-than-a-million-time-series).

    * Training models on Multiple Seasonalities.
        * Learn to use multiple seasonality in this [Electricity Load forecasting](../tutorials/electricityloadforecasting.html) tutorial.

    * Using external regressors or exogenous variables
        * Follow this tutorial to [include exogenous variables](../how-to-guides/exogenous.html) like weather or holidays or static variables like category or family.

    * Comparing StatsForecast with other popular libraries.
        * You can reproduce our benchmarks [here](https://github.com/Nixtla/statsforecast/tree/main/experiments).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install libraries

    We assume you have StatsForecast already installed. Check this guide for instructions on [how to install StatsForecast](./installation.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the data

    We will use polars to read the M4 Hourly data set stored in a parquet file for efficiency. You can use ordinary polars operations to read your data in other formats likes `.csv`.

    The input to StatsForecast is always a data frame in [long format](https://www.theanalysisfactor.com/wide-and-long-data/) with three columns: `unique_id`, `ds` and `y`:

    * The `unique_id` (string, int or category) represents an identifier for the series.

    * The `ds` (datestamp or int) column should be either an integer indexing time or a datestamp ideally like YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp.

    * The `y` (numeric) represents the measurement we wish to forecast.

    This data set already satisfies the requirement.

    Depending on your internet connection, this step should take around 10 seconds.
    """)
    return


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    Y_df = pl.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    Y_df.head()
    return (Y_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This dataset contains 414 unique series with 900 observations on average. For this example and reproducibility's sake, we will select only 10 unique IDs and keep only the last week. Depending on your processing infrastructure feel free to select more or less series.

    Processing time is dependent on the available computing resources. Running this example with the complete dataset takes around 10 minutes in a c5d.24xlarge (96 cores) instance from AWS.
    """)
    return


@app.cell
def _(Y_df, pl):
    uids = Y_df['unique_id'].unique(maintain_order=True)[:10]  # Select 10 ids to make the example faster
    Y_df_1 = Y_df.filter(pl.col('unique_id').is_in(uids))
    Y_df_1 = Y_df_1.group_by('unique_id').tail(7 * 24)  #Select last 7 days of data to make example faster
    return (Y_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Explore Data with the plot method
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot some series using the `plot` method from the `StatsForecast` class. This method prints 8 random series from the dataset and is useful for basic EDA.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `StatsForecast.plot` method uses matplotlib as a default engine. You can change to plotly by setting `engine="plotly"`.
    """)
    return


@app.cell
def _():
    from statsforecast import StatsForecast

    return (StatsForecast,)


@app.cell
def _(StatsForecast, Y_df_1):
    StatsForecast.plot(Y_df_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train multiple models for many series

    StatsForecast can train many models on many time series efficiently.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start by importing and instantiating the desired models. StatsForecast offers a wide variety of models grouped in the following categories:

    *  **Auto Forecast:** Automatic forecasting tools search for the best parameters and select the best possible model for a series of time series. These tools are useful for large collections of univariate time series. Includes automatic versions of: Arima, ETS, Theta, CES.

    * **Exponential Smoothing:**  Uses a weighted average of all past observations where the weights decrease exponentially into the past. Suitable for data with no clear trend or seasonality. Examples: SES, Holt's Winters, SSO.

    * **Benchmark models:** classical models for establishing baselines. Examples: Mean, Naive, Random Walk

    * **Intermittent or Sparse models:** suited for series with very few non-zero observations. Examples: CROSTON, ADIDA, IMAPA

    * **Multiple Seasonalities:** suited for signals with more than one clear seasonality. Useful for low-frequency data like electricity and logs. Examples: MSTL.

    * **Theta Models:**  fit two theta lines to a deseasonalized time series, using different techniques to obtain and combine the two theta lines to produce the final forecasts. Examples: Theta, DynamicTheta

    Here you can check the complete list of [models](../../src/core/models_intro.html).

    For this example we will use:

    * `AutoARIMA`: Automatically selects the best ARIMA (AutoRegressive Integrated Moving Average) model using an information criterion. Ref: `AutoARIMA`.

    * `HoltWinters`: triple exponential smoothing, Holt-Winters' method is an extension of exponential smoothing for series that contain both trend and seasonality. Ref: `HoltWinters`

    * `SeasonalNaive`: Memory Efficient Seasonal Naive predictions. Ref: `SeasonalNaive`

    * `HistoricAverage`: arithmetic mean. Ref: `HistoricAverage`.

    * `DynamicOptimizedTheta`: The theta family of models has been shown to perform well in various datasets such as M3. Models the deseasonalized time series. Ref: `DynamicOptimizedTheta`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Import and instantiate the models. Setting the `season_length` argument is sometimes tricky. This article on [Seasonal periods](https://robjhyndman.com/hyndsight/seasonal-periods/)) by the master, Rob Hyndmann, can be useful.
    """)
    return


@app.cell
def _():
    from statsforecast.models import (
        HoltWinters,
        CrostonClassic as Croston, 
        HistoricAverage,
        DynamicOptimizedTheta as DOT,
        SeasonalNaive
    )

    return Croston, DOT, HistoricAverage, HoltWinters, SeasonalNaive


@app.cell
def _(Croston, DOT, HistoricAverage, HoltWinters, SeasonalNaive):
    # Create a list of models and instantiation parameters
    models = [
        HoltWinters(),
        Croston(),
        SeasonalNaive(season_length=24),
        HistoricAverage(),
        DOT(season_length=24)
    ]
    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We fit the models by instantiating a new `StatsForecast` object with the following parameters:

    * `models`: a list of models. Select the models you want from [models](../../src/core/models_intro.html) and import them.

    * `freq`: a string indicating the frequency of the data. (See [panda's available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).) This is also available with Polars.

    * `n_jobs`: int, number of jobs used in the parallel processing, use -1 for all cores.

    * `fallback_model`: a model to be used if a model fails.

    Any settings are passed into the constructor. Then you call its fit method and pass in the historical data frame.
    """)
    return


@app.cell
def _(SeasonalNaive, StatsForecast, models):
    # Instantiate StatsForecast class as sf
    sf = StatsForecast( 
        models=models,
        freq=1, 
        n_jobs=-1,
        fallback_model=SeasonalNaive(season_length=7),
        verbose=True,
    )
    return (sf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `forecast` method takes two arguments: forecasts next `h` (horizon) and `level`.

    * `h` (int): represents the forecast h steps into the future. In this case, 12 months ahead.

    * `level` (list of floats): this optional parameter is used for probabilistic forecasting. Set the `level` (or confidence percentile) of your prediction interval. For example, `level=[90]` means that the model expects the real value to be inside that interval 90% of the times.

    The forecast object here is a new data frame that includes a column with the name of the model and the y hat values, as well as columns for the uncertainty intervals. Depending on your computer, this step should take around 1min. (If you want to speed things up to a couple of seconds, remove the AutoModels like ARIMA and Theta)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `forecast` method is compatible with distributed clusters, so it does not store any model parameters. If you want to store parameters for every model you can use the `fit` and `predict` methods. However, those methods are not defined for distributed engines like Spark, Ray or Dask.
    """)
    return


@app.cell
def _(Y_df_1, sf):
    forecasts_df = sf.forecast(df=Y_df_1, h=48, level=[90])
    forecasts_df.head()
    return (forecasts_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the results of 8 random series using the `StatsForecast.plot` method.
    """)
    return


@app.cell
def _(Y_df_1, forecasts_df, sf):
    sf.plot(Y_df_1, forecasts_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `StatsForecast.plot` allows for further customization. For example, plot the results of the different models and unique ids.
    """)
    return


@app.cell
def _(Y_df_1, forecasts_df, sf):
    # Plot to unique_ids and some selected models
    sf.plot(Y_df_1, forecasts_df, models=['HoltWinters', 'DynamicOptimizedTheta'], unique_ids=['H10', 'H105'], level=[90])
    return


@app.cell
def _(Y_df_1, forecasts_df, sf):
    # Explore other models 
    sf.plot(Y_df_1, forecasts_df, models=['SeasonalNaive'], unique_ids=['H10', 'H105'], level=[90])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate the model's performance

    In previous steps, we've taken our historical data to predict the future. However, to assess its accuracy we would also like to know how the model would have performed in the past. To assess the accuracy and robustness of your models on your data perform Cross-Validation.

    With time series data, **Cross Validation** is done by defining a sliding window across the historical data and predicting the period following it. This form of cross-validation allows us to arrive at a better estimation of our model's predictive abilities across a wider range of temporal instances while also keeping the data in the training set contiguous as is required by our models.

    The following graph depicts such a Cross Validation Strategy:

    ![](https://raw.githubusercontent.com/Nixtla/statsforecast/main/nbs/imgs/ChainedWindows.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cross-validation of time series models is considered a best practice but most implementations are very slow. The statsforecast library implements cross-validation as a distributed operation, making the process less time-consuming to perform. If you have big datasets you can also perform Cross Validation in a distributed cluster using Ray, Dask or Spark.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, we want to evaluate the performance of each model for the last 2 days (n_windows=2), forecasting every second day (step_size=48).  Depending on your computer, this step should take around 1 min.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Setting `n_windows=1` mirrors a traditional train-test split with our historical data serving as the training set and the last 48 hours serving as the testing set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `cross_validation` method from the `StatsForecast` class takes the following arguments.

    * `df`: training data frame

    * `h` (int): represents h steps into the future that are being forecasted. In this case, 24 hours ahead.

    * `step_size` (int): step size between each window. In other words: how often do you want to run the forecasting processes.

    * `n_windows`(int): number of windows used for cross validation. In other words: what number of forecasting processes in the past do you want to evaluate.
    """)
    return


@app.cell
def _(Y_df_1, sf):
    cv_df = sf.cross_validation(df=Y_df_1, h=24, step_size=24, n_windows=2)
    return (cv_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `cv_df` object is a new data frame that includes the following columns:

    * `unique_id`: series identifier

    * `ds`: datestamp or temporal index

    * `cutoff`: the last datestamp or temporal index for the `n_windows.` If `n_windows=1`, then one unique cutoff value, if `n_windows=2` then two unique cutoff values.

    * `y`: true value

    * `"model"`: columns with the model's name and fitted value.
    """)
    return


@app.cell
def _(cv_df):
    cv_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we will evaluate the performance of every model for every series using common error metrics like Mean Absolute Error (MAE) or Mean Square Error (MSE)
    Define a utility function to evaluate different error metrics for the cross validation data frame.

    First import the desired error metrics from `utilsforecast.losses`. Then define a utility function that takes a cross-validation data frame as a metric and returns an evaluation data frame with the average of the error metric for every unique id and fitted model and all cutoffs.
    """)
    return


@app.cell
def _():
    from utilsforecast.evaluation import evaluate
    from utilsforecast.losses import mse

    return evaluate, mse


@app.cell
def _(evaluate, pl):
    def evaluate_cv(df, metric):
        models = [c for c in df.columns if c not in ('unique_id', 'ds', 'cutoff', 'y')]
        evals = evaluate(df, metrics=[metric], models=models)
        evals = evals.drop('metric')
        pos2model = dict(enumerate(models))
        return evals.with_columns(
            best_model=pl.concat_list(models).list.arg_min().replace_strict(pos2model)
        )

    return (evaluate_cv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use Mean Average Percentage Error (MAPE), however for granular forecasts, MAPE values are extremely [hard to judge](https://blog.blueyonder.com/mean-absolute-percentage-error-mape-has-served-its-duty-and-should-now-retire/) and not useful to assess forecasting quality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create the data frame with the results of the evaluation of your cross-validation data frame using a Mean Squared Error metric.
    """)
    return


@app.cell
def _(cv_df, evaluate_cv, mse):
    evaluation_df = evaluate_cv(cv_df, mse)
    evaluation_df.head()
    return (evaluation_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a summary table with a model column and the number of series where that model performs best. In this case, the Arima and Seasonal Naive are the best models for 10 series and the Theta model should be used for two.
    """)
    return


@app.cell
def _(evaluation_df):
    evaluation_df['best_model'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can further explore your results by plotting the unique_ids where a specific model wins.
    """)
    return


@app.cell
def _(Y_df_1, evaluation_df, forecasts_df, pl, sf):
    seasonal_ids = evaluation_df.filter(pl.col('best_model') == 'SeasonalNaive')['unique_id']
    sf.plot(Y_df_1, forecasts_df, unique_ids=seasonal_ids, models=['SeasonalNaive', 'DynamicOptimizedTheta'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select the best model for every unique series
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define a utility function that takes your forecast's data frame with the predictions and the evaluation data frame and returns a data frame with the best possible forecast for every unique_id.
    """)
    return


@app.cell
def _(pl):
    def get_best_model_forecast(forecasts_df, evaluation_df):
        models = {
            c.replace('-lo-90', '').replace('-hi-90', '')
            for c in forecasts_df.columns
            if c not in ('unique_id', 'ds')
        }
        model2pos = {m: i for i, m in enumerate(models)}
        with_best = forecasts_df.join(evaluation_df[['unique_id', 'best_model']], on='unique_id')
        return with_best.select(
            'unique_id',
            'ds',
            *[
                (
                    pl.concat_list([f'{m}{suffix}' for m in models])
                    .list.get(pl.col('best_model').replace_strict(model2pos))
                    .alias(f'best_model{suffix}')
                )
                for suffix in ('', '-lo-90', '-hi-90')
            ]
        )

    return (get_best_model_forecast,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create your production-ready data frame with the best forecast for every unique_id.
    """)
    return


@app.cell
def _(evaluation_df, forecasts_df, get_best_model_forecast):
    prod_forecasts_df = get_best_model_forecast(forecasts_df, evaluation_df)
    prod_forecasts_df.head()
    return (prod_forecasts_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plot the results.
    """)
    return


@app.cell
def _(Y_df_1, prod_forecasts_df, sf):
    sf.plot(Y_df_1, prod_forecasts_df, level=[90])
    return


if __name__ == "__main__":
    app.run()
