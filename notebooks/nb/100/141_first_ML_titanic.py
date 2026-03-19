import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import polars as pl
    import altair as alt

    from pathlib import Path


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Titanic Survival Prediction
    """)
    return


@app.cell
def _():
    TITANIC_OUT_FILE = Path("data/titanic/titanic.parquet")
    TITANIC_META_FILE = TITANIC_OUT_FILE.with_suffix(".meta.json")

    # Read dataframe
    df = pl.read_parquet(TITANIC_OUT_FILE)

    # Read metadata
    with open(TITANIC_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    CATEGORICALS = meta["categoricals"]
    NUMERICS = meta["numerics"]
    DROP_COLS = meta["drop_cols"]
    TARGET = meta["target"]

    print("[INFO] Data columns:", df.columns)
    print("[INFO] CATEGORICALS:", CATEGORICALS)
    print("[INFO] NUMERICS:", NUMERICS)
    print("[INFO] DROP_COLS:", DROP_COLS)
    print("[INFO] TARGET:", TARGET)

    # Typical split
    X = df.drop(TARGET)
    y = df.select(TARGET)
    return (df,)


@app.cell
def _(df):
    from skrub import TableReport

    TableReport(df)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
