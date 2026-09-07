import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    import json
    import polars as pl
    import altair as alt
    import numpy as np

    from pathlib import Path

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        roc_curve,
        average_precision_score,
        balanced_accuracy_score
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ySKILLS Cyberhate Risk Prediction

    Let's start by loading the semantic Parquet file produced by `713_y_skills_EDA.py`, plus its sidecar metadata. If you do not have the file, either follow guide in the mentioned Notebook, or use the zipped data provided by the teacher. To do that, run...

    ```
    cd notebooks
    unzip gitlfs-store/yskills.zip
    ```

    ## Data Loading
    """)
    return


@app.cell
def _():
    YSKILLS_OUT_FILE = Path("data/y_skills/y_skills_model_independent_transformations.parquet")
    YSKILLS_META_FILE = YSKILLS_OUT_FILE.with_suffix(".meta.json")

    # Read dataframe
    df = pl.read_parquet(YSKILLS_OUT_FILE)

    # Read metadata
    with open(YSKILLS_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    CATEGORICALS = meta["categoricals"]
    NUMERICS = meta["numerics"]
    TARGET = meta["target"][0]

    print("[INFO] Data columns:", df.columns)
    print("[INFO] CATEGORICALS:", CATEGORICALS)
    print("[INFO] NUMERICS:", NUMERICS)
    print("[INFO] TARGET:", TARGET)
    return CATEGORICALS, NUMERICS, TARGET, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Test Split

    `RISK101` is imbalanced (about 72% of respondents report past-year cyberhate experience), so we stratify the split on the target.
    """)
    return


@app.cell
def _(TARGET, df):
    X = df.drop(TARGET)
    y = df.get_column(TARGET)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[INFO] Lines in training set", X_train.height)
    print("[INFO] Lines in testing set", X_test.height)
    print("[INFO] Target balance in training set:", y_train.value_counts())

    X_train
    return X_test, X_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One Hot Encoding

    This encoder CAN be placed into the Pipeline further down, but let's leave it here, pre-fit on the training data only, to avoid leakage.
    """)
    return


@app.cell
def _(CATEGORICALS, NUMERICS, X_test, X_train):
    def one_hot_encode(X):
        encoded = ohe.transform(X.select(CATEGORICALS))
        return pl.concat([
            X.select(NUMERICS),
            pl.DataFrame(encoded, schema=ohe_feature_names)
        ], how="horizontal_extend")

    # Pre-fit OneHotEncoder on training data only, outside of GridSearch
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop="if_binary")
    ohe.fit(X_train.select(CATEGORICALS))

    ohe_feature_names = list(ohe.get_feature_names_out(CATEGORICALS))

    X_train_encoded = one_hot_encode(X_train)
    X_test_encoded = one_hot_encode(X_test)

    print("[INFO] Encoded training set shape:", X_train_encoded.shape)
    print("[INFO] OHE feature names:")
    for feat_name in ohe_feature_names:
        print("  - ", feat_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # (Your work starts at:) Training

    You might want to play around with both `LogisticRegression` and `SGDClassifier()`, and maybe use GridCVSearch instead of manually testing -- but that works too!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Coefficients

    Both `LogisticRegression` and `SGDClassifier(loss="log_loss")` fit the same linear-in-log-odds model, so their coefficients are directly comparable.
    """)
    return


if __name__ == "__main__":
    app.run()
