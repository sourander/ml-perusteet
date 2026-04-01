import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt
    import polars.selectors as cs

    from pathlib import Path
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn import metrics
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automatic Transmissions

    In this Notebook, you train train a Decision Tree and Random Forest models on the given data. You should compare the models. The task is **binary classification**, so you will need to re-label the target variable into a binary form. The original values and their potential mapped counterparts are...

    | Original value (str)    | New value (bool) | N         |
    | ----------------------- | ---------------- | --------- |
    | AUTOMATIC               | 1                | 8266      |
    | MANUAL                  | 0                | 2935      |
    | AUTOMATED_MANUAL        | 0/1              | 626       |
    | DIRECT_DRIVE            | (drop)           | 68        |
    | UNKNOWN                 | (drop)           | 19        |

    /// note
    If you are a car expert, you can process the `DIRECT_DRIVE` and `UNKNOWN` values. However, for this exercise, you can also simply drop them. Remember to document and explain your choices; especially if they differ from the assignment brief.
    ///

    ## Data

    You will need to download the data. You can download it from multiple places, including but not limited to...

    * [gh:suhasmaddali/Car-Prices-Prediction](https://github.com/suhasmaddali/Car-Prices-Prediction/tree/main)
    * [Kaggle: CooperUnion | Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset)
    """)
    return


@app.cell
def _():
    CARS_SCHEMA = {
        "Make": pl.String,                # e.g. BMW
        "Model": pl.String,               # e.g. 1 Series
        "Year": pl.Int16,                 # e.g. 2011
        "Engine Fuel Type": pl.String,    # e.g. premium unleaded (required)
        "Engine HP": pl.Int16,            # horsepower
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
        "MSRP": pl.Int32,                 # price in dollars
    }


    transmission_map = {
        "AUTOMATIC": 1,
        "MANUAL": 0,
        "AUTOMATED_MANUAL": None,  # <- This is something you need to choose!
        "DIRECT_DRIVE": None,
        "UNKNOWN": None,
    }


    df = pl.read_csv(
        Path("data/car/data.csv"),
        columns=list(CARS_SCHEMA.keys()),
        schema_overrides=CARS_SCHEMA
    ).with_columns(
        pl.col("Transmission Type")
        .replace_strict(transmission_map, default=None, return_dtype=pl.Int8)
        .cast(pl.Int8)
        .alias("is_automatic")
    ).filter(
        pl.col("is_automatic").is_not_null()
    ).drop("Transmission Type")

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quick n Dirty EDA

    This is the very minimum EDA that is required to do any decisions that make sense. Of course, feel free to do a lot more.

    ### Describe cols
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check Label Imbalance
    """)
    return


@app.cell(hide_code=True)
def _(df):
    # replace _df with your data source
    _chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(field='is_automatic', type='nominal'),
            y=alt.Y(aggregate='count', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='label', format=',.0f'),
                alt.Tooltip(aggregate='count')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Uniques
    """)
    return


@app.cell
def _(df):
    MAX_UNIQUES = 16

    for col in df.columns:
        uniques = df.get_column(col).unique().sort().to_list()

        shown = uniques[:MAX_UNIQUES]
        total = len(uniques)

        print(f"\n{col} ({total} unique):")
        for value in shown:
            print(f"  {value!r}")

        if total > MAX_UNIQUES:
            print(f"  ... and {total - MAX_UNIQUES} more")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scatter Plot

    To visualize how the columns correlate, a pair plot is nice - if your dataset is small and you don't have too many columns. You can try this with this dataset, but you might not like what you're seeing.
    """)
    return


@app.cell(hide_code=True)
def _(df):
    _continuous_values = ["Year", "Engine HP", "highway MPG", "city mpg", "Popularity", "MSRP"]

    # The MSRP has a couple of outliers that make the diagram harder to read.
    # This will cap car price to 500k
    _df_plot = df.with_columns(
        pl.col("MSRP").clip(upper_bound=500_000)
    )

    _chart = alt.Chart(_df_plot).mark_circle().encode(
        x=alt.X(
            alt.repeat("column"),
            type="quantitative",
            scale=alt.Scale(zero=False)
        ),
        y=alt.Y(
            alt.repeat("row"),
            type="quantitative",
            scale=alt.Scale(zero=False)
        ),
        color=alt.Color("is_automatic:N", title="Automatic")
    ).properties(
        width=125,
        height=125
    ).repeat(
        row=_continuous_values,
        column=_continuous_values
    ).resolve_scale(
        x="independent",
        y="independent"
    )

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Correlation Matrix

    We will discuss correlation on the upcoming lessons more (relating to e.g. linear regression)
    """)
    return


@app.cell(hide_code=True)
def _(df):
    _numeric_cols = list(df.select(cs.numeric()).columns)

    # Compute correlation matrix in Polars
    corr_df = df.select(_numeric_cols).corr()

    # Add row labels (because corr() returns a square matrix without row names)
    corr_df = corr_df.with_columns(pl.Series("feature_y", _numeric_cols))

    # Reshape to long format for Altair
    corr_long = corr_df.unpivot(
        index="feature_y",
        variable_name="feature_x",
        value_name="correlation"
    )

    _chart = alt.Chart(corr_long).mark_rect().encode(
        x=alt.X("feature_x:N", title=None),
        y=alt.Y("feature_y:N", title=None),
        color=alt.Color(
            "correlation:Q",
            scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
            title="Correlation"
        ),
        tooltip=[
            alt.Tooltip("feature_x:N", title="X"),
            alt.Tooltip("feature_y:N", title="Y"),
            alt.Tooltip("correlation:Q", format=".3f")
        ]
    ).properties(
        width=400,
        height=400,
        title="Correlation Heatmap"
    )

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Invididual Column Plots

    If you'd do a proper EDA, you'd have to investigate each invididually with care.

    To give you a starting point, I have given a quick 'n' dirty way of selecting a column and creating multiple plots of that column. They might fit the task, they might not. Edit the code as you please.
    """)
    return


@app.cell(hide_code=True)
def _(df, mo):
    _numerical_columns = list(df.select(cs.numeric()).columns)
    _n = len(_numerical_columns) - 1

    year_selector = mo.ui.number(start=0, stop=_n)
    year_selector
    return (year_selector,)


@app.cell(hide_code=True)
def _(df, year_selector):
    NUMERICAL_COLS_plotting = list(df.select(cs.numeric()).columns)
    CHOSEN_plotting = NUMERICAL_COLS_plotting[year_selector.value]

    _chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{CHOSEN_plotting}:O", title=CHOSEN_plotting),
        y=alt.Y("count()", stack="normalize", title="Proportion"),
        color=alt.Color("is_automatic:N", title="Automatic")
    ).properties(width="container")

    _chart
    return (CHOSEN_plotting,)


@app.cell(hide_code=True)
def _(CHOSEN_plotting, df):
    _chart = alt.Chart(df).mark_bar(opacity=0.6).encode(
        x=alt.X(
            f"{CHOSEN_plotting}:Q",
            bin=alt.Bin(maxbins=30),
            title=CHOSEN_plotting
        ),
        y=alt.Y("count()", title="Count"),
        color=alt.Color("is_automatic:N", title="Automatic")
    ).properties(width="container")

    _chart2 = alt.Chart(df).mark_boxplot().encode(
        x=alt.X("is_automatic:N", title="Automatic"),
        y=alt.Y(f"{CHOSEN_plotting}:Q", title=CHOSEN_plotting)
    ).properties(width="container")

    _chart | _chart2 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check MSRP too

    You probably noticed that the MSRP behaves badly when plotting. You could try to bin it using a histogram, but even if you cap the high prices, you would get the following results....
    """)
    return


@app.cell(hide_code=True)
def _(df):
    _chart = alt.Chart(
        df.with_columns(pl.col("MSRP").clip(upper_bound=300_000))
    ).mark_bar().encode(
        x=alt.X(
            "MSRP:Q",
            bin=alt.Bin(maxbins=50),
            title="MSRP"
        ),
        y="count()",
        color="is_automatic:N"
    ).properties(width="container")

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Applying a log to the price will make inspection easier.
    """)
    return


@app.cell(hide_code=True)
def _(df):
    chart = alt.Chart(
        df.with_columns(pl.col("MSRP").log10().alias("log10_MSRP"))
    ).mark_bar().encode(
        x=alt.X("log10_MSRP:Q", bin=alt.Bin(maxbins=30), title="log10(MSRP)"),
        y="count()",
        color="is_automatic:N"
    ).properties(width="container")

    chart
    return


@app.cell
def _():
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
    TARGET = "is_automatic"
    LEAKAGE_COLS = ["Transmission Type"]  # original source of the label -> must be dropped

    # Numeric feature columns (excluding target)
    numeric_features = list(df.select(cs.numeric()).columns)
    numeric_features.remove(TARGET)

    # All usable feature columns
    feature_cols = [
        c for c in df.columns
        if c not in [TARGET] + LEAKAGE_COLS
    ]

    # Categorical features = everything in feature_cols that is not numeric
    categorical_features = [
        c for c in feature_cols
        if c not in numeric_features
    ]

    print("Numeric features:")
    print(numeric_features)
    print()

    print("Categorical features:")
    print(categorical_features)
    print()

    print(f"Number of features: {len(feature_cols)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create X and y
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Test Split
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing pipeline
    """)
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
    # Inspect Feature Importance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot (Decision) Tree
    """)
    return


if __name__ == "__main__":
    app.run()
