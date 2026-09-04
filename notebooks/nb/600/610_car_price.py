import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt
    import polars.selectors as cs

    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn import metrics


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Predicting Car Prices (MSRP)

    This Notebook trains a **Linear Regression** model to predict a car's manufacturer suggested retail price (`MSRP`). It reuses the same dataset as `331_automatic_transmission.py` and `421_automatic_transmission_knn.py`, but this time we are not classifying transmission type. We are predicting a continuous target, which is the task the original dataset was built for in the first place.

    We already covered the dataset basics in the previous notebooks (`describe()`, unique-value listings, a pairwise scatter matrix, a correlation heatmap and the skewed `MSRP` distribution including its `log10` transform). Go check that Notebook if you need a refresher.

    Metrics of interest: **MSE**, **RMSE**, **MAE** and **R²**. We also compare a plain Linear Regression against one enriched with `PolynomialFeatures`, to see whether the extra complexity helps or just overfits.
    """)
    return


@app.cell
def _():
    CARS_SCHEMA = {
        "Make": pl.String,
        "Model": pl.String,
        "Year": pl.Int16,
        "Engine Fuel Type": pl.String,
        "Engine HP": pl.Int16,
        "Engine Cylinders": pl.Int8,
        "Transmission Type": pl.String,
        "Driven_Wheels": pl.String,
        "Number of Doors": pl.Int8,
        "Market Category": pl.String,
        "Vehicle Size": pl.String,
        "Vehicle Style": pl.String,
        "highway MPG": pl.Int16,
        "city mpg": pl.Int16,
        "Popularity": pl.Int16,
        "MSRP": pl.Int32,
    }

    df = pl.read_csv(
        Path("data/car/data.csv"),
        columns=list(CARS_SCHEMA.keys()),
        schema_overrides=CARS_SCHEMA,
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis

    We have already performed some EDA on the dataset, there here are a few new approches that are useful specifically before fitting a regression model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Missing Values
    """)
    return


@app.cell(hide_code=True)
def _(df):
    _null_counts = df.null_count().transpose(
        include_header=True, header_name="column", column_names=["nulls"]
    )

    _chart = (
        alt.Chart(_null_counts)
        .mark_bar()
        .encode(
            x=alt.X("nulls:Q", title="Missing values"),
            y=alt.Y("column:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("column:N"), alt.Tooltip("nulls:Q")],
        )
        .properties(width="container", height=320)
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Average Price by Make
    """)
    return


@app.cell(hide_code=True)
def _(df):
    _avg_price_by_make = (
        df.group_by("Make")
        .agg(pl.col("MSRP").mean().alias("avg_MSRP"))
        .sort("avg_MSRP", descending=True)
        .head(20)
    )

    _chart = (
        alt.Chart(_avg_price_by_make)
        .mark_bar()
        .encode(
            x=alt.X("avg_MSRP:Q", title="Average MSRP"),
            y=alt.Y("Make:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Make:N"),
                alt.Tooltip("avg_MSRP:Q", title="Average MSRP", format=",.0f"),
            ],
        )
        .properties(width="container", height=400)
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature vs MSRP

    A quick look at how well a single feature already explains the price, with a linear trend fitted through it. Pick a column below and optionally drop the outliers above 1,000,000 MSRP.
    """)
    return


@app.cell
def _(df, mo):
    _numeric_cols = [c for c in df.select(cs.numeric()).columns if c != "MSRP"]

    column_dd = mo.ui.dropdown(
        options=_numeric_cols,
        value="Engine HP",
        label="Column vs. MSRP",
    )

    drop_outliers_cb = mo.ui.checkbox(
        value=False,
        label="Drop outliers (MSRP > 1,000,000)",
    )

    mo.hstack([column_dd, drop_outliers_cb])
    return column_dd, drop_outliers_cb


@app.cell(hide_code=True)
def _(column_dd, df, drop_outliers_cb):
    _plot_df = df.filter(pl.col("MSRP") <= 1_000_000) if drop_outliers_cb.value else df

    _scatter = (
        alt.Chart(_plot_df)
        .mark_circle(opacity=0.25)
        .encode(
            x=alt.X(f"{column_dd.value}:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("MSRP:Q", scale=alt.Scale(zero=False)),
        )
    )

    _trend = _scatter.transform_regression(column_dd.value, "MSRP").mark_line(color="red")

    _chart = (_scatter + _trend).properties(width="container", height=350)
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Features

    ## Define Feature Columns
    """)
    return


@app.cell
def _(df):
    TARGET = "MSRP"

    numeric_features = list(df.select(cs.numeric()).columns)
    numeric_features.remove(TARGET)

    feature_cols = [c for c in df.columns if c != TARGET]

    categorical_features = [c for c in feature_cols if c not in numeric_features]

    print("Numeric features:")
    print(numeric_features)
    print()

    print("Categorical features:")
    print(categorical_features)
    print()

    print(f"Number of features: {len(feature_cols)}")
    return TARGET, feature_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create X and y
    """)
    return


@app.cell
def _(TARGET, df, feature_cols):
    X = df.select(feature_cols)
    y = df.get_column(TARGET)
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Test Split
    """)
    return


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Explainability

    Unlike black-box models, a Linear Regression is directly explainable through its coefficients -- no SHAP required. Because the numeric features were standardized, coefficient magnitude roughly reflects how much one standard deviation of a feature moves the predicted price.

    ## Coefficients (Linear Regression)
    """)
    return


if __name__ == "__main__":
    app.run()
