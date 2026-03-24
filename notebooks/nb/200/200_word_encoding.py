import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Categorical String Encoding
    """)
    return


@app.cell
def _():
    data = {
        "age": [21, 34, 28, 45, 23, 39, 31, 26, 42, 29],
        "level": [
            "matala",
            "korkea",
            "keskitaso",
            "korkea",
            "matala",
            "keskitaso",
            "korkea",
            "matala",
            "keskitaso",
            "matala",
        ],
        "lang": [
            "Python",
            "R",
            "Python",
            "Julia",
            "R",
            "Python",
            "Julia",
            "Python",
            "R",
            "Julia",
        ],
        "feedback": [
            "selkeä teoria helppo selkeä tehtävä",
            "vaikea tehtävä paljon laskentaa",
            "selkeä harjoitus hyödyllinen esimerkki",
            "raskas projekti paljon koodia",
            "helppo harjoitus selkeä ohje",
            "hyödyllinen projekti hyvä visualisointi",
            "vaikea teoria raskas tehtävä",
            "selkeä esimerkki helppo koodi",
            "paljon teoria hyödyllinen tehtävä",
            "helppo alku selkeä harjoitus",
        ],
        "passed": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    }

    df = pl.DataFrame(
        data,
        schema={
            "age": pl.Int8,
            "level": pl.Utf8,
            "lang": pl.Utf8,
            "feedback": pl.Utf8,
            "passed": pl.Int8,
        }
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One-Hot Encoding

    ### Using SQL
    """)
    return


@app.cell
def _(df, mo):
    _df = mo.sql(
        f"""
        SELECT
            "level",
            level_keskitaso: ("level" = 'keskitaso')::UTINYINT,
            level_korkea: ("level" = 'korkea')::UTINYINT,
            level_matala: ("level" = 'matala')::UTINYINT
        FROM df
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using Polars
    """)
    return


@app.cell
def _(df):
    df.select("level").hstack(df.select("level").to_dummies())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Label Encoding


    ...or, Ordinal Encoding with an arbitrary order.

    For understanding the code, you might want to check the Polars docs: [polars.Series.to_physical](https://docs.pola.rs/api/python/dev/reference/series/api/polars.Series.to_physical.html)
    """)
    return


@app.cell
def _(df):
    df.with_columns(
            pl.col("level")
            .cast(pl.Categorical)
            .to_physical()
            .alias("level_encoded")
    ).select("level", "level_encoded")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ordinal Encoding
    """)
    return


@app.cell
def _(df):
    # Define the explicit order for ordinal categories
    level_order = ["matala", "keskitaso", "korkea"]

    # Create a mapping dict: {"matala":0, "keskitaso":1, "korkea":2}
    level_mapping = {level: i for i, level in enumerate(level_order)}


    df.with_columns(
        pl.col("level").replace(level_mapping).alias("level_encoded")
    ).select("level", "level_encoded")
    return


if __name__ == "__main__":
    app.run()
