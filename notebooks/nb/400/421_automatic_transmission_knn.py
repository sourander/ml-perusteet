import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import polars as pl
    import altair as alt
    import polars.selectors as cs

    from pathlib import Path
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import PCA
    from sklearn import metrics
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    # Automatic Transmissions (k-NN)

    In this Notebook, you train **k-Nearest Neighbours (k-NN)** classifiers on car data to
    predict whether a car has an automatic or manual transmission.

    The teacher's Random Forest obtained a 96-97 % F1 score using the hold-out test set. What was your Random Forest and/or Decision Tree score? Try to get the k-NN to match that.

    Is it suggested to compare **multiple variants** to understand the impact of feature engineering and
    dimensionality reduction on a distance-based model. For example, you could create Models A, B and C, using...

    | Model | Features | PCA |
    | ----- | -------- | --- |
    | **A** | Numeric + categorical *without* `Make`/`Model` | No |
    | **B** | All features (full OHE incl. `Make`/`Model`) | No |
    | **C** | All features (full OHE incl. `Make`/`Model`) | Yes |

    Then, use either GridSearchCV or a manual train/val split to find the best hyperparameters for each. One key hyperparameter is the value of $k$.

    ## Target variable

    The dataset is the same as before, and you should use the same data pre-processing so that the models are comparable. For recap, the target field will be binarized version of...

    | Original value (str)    | New value (int) | N         |
    | ----------------------- | --------------- | --------- |
    | AUTOMATIC               | 1               | 8266      |
    | MANUAL                  | 0               | 2935      |
    | AUTOMATED_MANUAL        | 0/1             | 626       |
    | DIRECT_DRIVE            | (drop)          | 68        |
    | UNKNOWN                 | (drop)          | 19        |

    ## Data

    You should already have the data downloaded.
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

    df = df.with_columns(pl.col("MSRP").log(10))

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quick n Dirty EDA

    This work was already done during the Decision Tree and Random Forest training. Refer to that Notebook if needed (`331_automatic_transmission.py`)
    """)
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
    LEAKAGE_COLS = [
        "Transmission Type"
    ]  # original source of the label -> must be dropped

    # Numeric feature columns (excluding target)
    numeric_features = list(df.select(cs.numeric()).columns)
    numeric_features.remove(TARGET)

    # All usable categorical features (everything non-numeric, non-target)
    _all_cols = [c for c in df.columns if c not in [TARGET] + LEAKAGE_COLS]
    categorical_features_full = [c for c in _all_cols if c not in numeric_features]

    # Simplified: drop high-cardinality Make and Model (used in Model A)
    categorical_features_simple = [
        c for c in categorical_features_full if c not in ("Make", "Model")
    ]

    feature_cols_full = numeric_features + categorical_features_full
    feature_cols_simple = numeric_features + categorical_features_simple

    print("Numeric features:")
    print(numeric_features)
    print()

    print("Categorical features (full):")
    print(categorical_features_full)
    print()

    print("Categorical features (simple, no Make/Model):")
    print(categorical_features_simple)
    print()

    print(
        f"Feature count — full: {len(feature_cols_full)},  simple: {len(feature_cols_simple)}"
    )
    return TARGET, feature_cols_full, feature_cols_simple


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create X and y
    """)
    return


@app.cell
def _(TARGET, df, feature_cols_full, feature_cols_simple):
    X_full = df.select(feature_cols_full)
    X_simple = df.select(feature_cols_simple) # Simple lacks the Model and Make
    y = df.get_column(TARGET)
    return X_full, X_simple, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Test Split
    """)
    return


@app.cell
def _(X_full, X_simple, y):
    # Split all arrays simultaneously to guarantee identical train/test row sets.
    X_full_train, X_full_test, X_simple_train, X_simple_test, y_train, y_test = (
        train_test_split(
            X_full, X_simple, y,
            test_size=0.20,
            stratify=y,
            random_state=42,
        )
    )

    print("Train shape (full)  :", X_full_train.shape)
    print("Test  shape (full)  :", X_full_test.shape)
    print()

    print("Train target distribution:")
    print(y_train.value_counts(normalize=True).sort("is_automatic"))
    print()

    print("Test target distribution:")
    print(y_test.value_counts(normalize=True).sort("is_automatic"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing Pipelines
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Configure Search
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model A — k-NN (Simple Features, No PCA)

    Uses numeric features plus all categorical features **except** `Make` and `Model`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model B — k-NN (Full OHE, No PCA)

    Uses **all** features including `Make` and `Model`, which are One-Hot Encoded.
    This produces hundreds of binary columns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model C — k-NN (Full OHE + PCA)

    Same full feature set as Model B, but **PCA** is applied after OHE to reduce
    dimensionality before k-NN sees the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Comparison
    """)
    return


if __name__ == "__main__":
    app.run()
