import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt
    import polars.selectors as cs
    import json

    from pathlib import Path


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # First EDA Ever (with Titanic)

    > "In statistics, exploratory data analysis (EDA) or exploratory analytics is an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods."
    >
    > – Wikipedia, https://en.wikipedia.org/wiki/Exploratory_data_analysis

    In this document, we will inspect the Titanic dataset column by column and decide what belongs in a semantic feature layer – which is a fancy way of saying that we will preprocess the data and save it to disk, and continue in another script. This notebook also works as a smoke test: if you can run this, your environment has been set up correctly. If not, ask for help. To start, you need to unzip the data files from the Git LFS store into the local `data/` directory. You can do this like so...

    ```bash
    cd notebooks
    unzip gitlfs-store/titanic.zip
    ```

    ## Goal of this file

    We want to produce a semantic Parquet artifact at `data/titanic/titanic.parquet`. In this notebook, **semantic** means: transforms that are deterministic, justified from the meaning of the raw column, and do not require fitting on training data.

    This notebook is responsible for:

    * inspecting columns
    * ...and choosing what to do (verdicts)
    * applying (deterministic) feature extraction
    * standardizing nulls
    * dropping obvious leakage columns
    * saving a semantic Parquet plus sidecar metadata

    Good examples of work that belongs here are parsing titles from names, deriving `family` and `is_alone`, extracting `ticketprefix` and `cabinprefix`, and resolving values when domain knowledge gives a deterministic answer.

    Anything that is fit (=learned) from data, depends on the downstream estimator (=ML model), or acts like a tunable hyperparameter must be left for the next notebook. That includes:

    * imputing unresolved missing values
    * one-hot encoding or ordinal encoding
    * scaling numeric features
    * rare-category bucketing

    The next notebook will load this semantic Parquet, split `X` and `y`, and continue from there.
    """)
    return


@app.cell
def _():
    TITANIC_FILE = Path("data/titanic/titanic.csv")
    TITANIC_OUT_FILE = Path("data/titanic/titanic.parquet")
    TITANIC_META_FILE = TITANIC_OUT_FILE.with_suffix(".meta.json")

    TITANIC_SCHEMA = {
        "pclass": pl.Int8,       # only values 1, 2, 3. Ordinal.
        "survived": pl.Int8,     # only values 0 and 1. Ordinal.
        "name": pl.String,
        "sex": pl.String,
        "age": pl.Float32,       # babies are with decimal value
        "sibsp": pl.Int8,        # integer values 0 to 8
        "parch": pl.Int8,        # integer values 0 to 9
        "ticket": pl.String,     # not a number; but (high cardinality) categorical data
        "fare": pl.Float32,
        "cabin": pl.String,
        "embarked": pl.String,
        "boat": pl.String,
        "body": pl.Int32,
        "home.dest": pl.String,
    }

    # Note that these must be found manually. An educated guess is
    # that ["", "NA", "N/A", "null", "?"] may be null values, but a CSV might
    # as well have nulls like ["missing", "MISSING", "NONE", "NULL"]. These
    # will be found during EDA process.
    TITANIC_NULL_VALUES = ["", "NA", "N/A", "null", "?"]
    return (
        TITANIC_FILE,
        TITANIC_META_FILE,
        TITANIC_NULL_VALUES,
        TITANIC_OUT_FILE,
        TITANIC_SCHEMA,
    )


@app.cell
def _():
    # This is how you would start to investigate the value ranges:

    # df_inferred = pl.read_csv(TITANIC_FILE)
    # df_inferred.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What is schema?

    A simple CSV does not contain any *schema information* (=the data types of each columns). Only information that can be read from the file is what is the name of the field (if header exists) and whether the field contains a number, a string or a missing value.

    For example, it makes sense to read small integer values as `INT8`, taking only 8 bits instead the default `f64`, which takes 64 bits of memory. This does not simply just save our RAM (by storing less bits per value), but also makes it easier to visualize and process the data later on.

    The data has been originally downloaded from ~~`http://biostat.mc.vanderbilt.edu/DataSets`~~  and that site is no longer online. You can find the metadata at [TitanicMETA.pdf](http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf), but the table description is replicated below as Markdown table for your convenience.


    | Field      | Description                                                             |
    |------------|:------------------------------------------------------------------------|
    | pclass     | Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)                             |
    | survival   | Survival (0 = No; 1 = Yes)                                              |
    | name       | Name                                                                    |
    | sex        | Sex                                                                     |
    | age        | Age                                                                     |
    | sibsp      | Number of Siblings/Spouses Aboard                                       |
    | parch      | Number of Parents/Children Aboard                                       |
    | ticket     | Ticket Number                                                           |
    | fare       | Passenger Fare (British pound)                                          |
    | cabin      | Cabin                                                                   |
    | embarked   | Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)    |
    | boat       | Lifeboat                                                                |
    | body       | Body Identification Number                                              |
    | home.dest  | Home/Destination                                                        |
    """)
    return


@app.cell
def _(TITANIC_FILE, TITANIC_NULL_VALUES, TITANIC_SCHEMA):
    df = pl.read_csv(
        TITANIC_FILE,
        schema_overrides=TITANIC_SCHEMA,
        null_values=TITANIC_NULL_VALUES,
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset overview

    The whole dataset can be checked at once. There are multiple ways to do this, including also...

    ```python
    from skrub import TableReport
    TableReport(df)
    ```

    The code block above would run nearly 1 minute and print an interactive report you can investigate. Here we will simply check a typical **scatter matrix** to get a quick feel for the numeric columns.

    We intentionally leave out `body` from this overview even though it is numeric, because it is an obvious leakage column for survival prediction and would distract from the rest of the inspection.
    """)
    return


@app.cell
def _(df):
    # Keep all numeric columns except the survived
    _numeric_cols = list(df.select(cs.numeric()).columns)
    _numeric_cols.remove("survived")

    # Try it yourself: check what the output of this looks like 
    # by commenting out the next row
    _numeric_cols.remove("body") 

    alt.Chart(df).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color=alt.Color("survived:N", title="Survived")
    ).properties(
        width=125,
        height=125
    ).repeat(
        row=_numeric_cols + ["survived"],
        column=_numeric_cols
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Column-level work

    We will go through these 14 columns one by one. For each column, we will place it into one of four buckets:

    * keep as numeric
    * keep as categorical
    * derive one or more features (and drop original)
    * drop

    The rule is simple: if a decision can be made deterministically from the raw value and domain meaning, it belongs here. If it would need training data, frequency statistics, estimator assumptions, or hyperparameter choices, it belongs in the next notebook.
    """)
    return


@app.cell
def _(df):
    for i, col_name in enumerate(df.columns, start=1):
        print(f"{i}: {col_name}")


    CATEGORICALS = set()
    NUMERICS = set()
    DROP_COLS = set()

    TARGET = ["survived"]
    return CATEGORICALS, DROP_COLS, NUMERICS, TARGET


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Columns

    ## 1. pclass

    Note that the `df.plot` uses Altair internally. This is exactly same as running...

    ```
    alt.Chart(df).mark_bar().encode(
        x='pclass:N',
        y='survived'
    )
    ```

    `pclass` is stored as a small integer, but semantically it acts like a low-cardinality category: first, second, or third class.

    **Verdict:** Keep it as a categorical feature. We are not ordinal-encoding it here; we are only deciding that this field should survive into the semantic dataset.
    """)
    return


@app.function
def print_null_and_uniques(colname, dataframe, up_to=30):
    print(f"[INFO] The {colname} has null values: ", dataframe.select(colname).null_count().item())
    print(f"[INFO] Unique values has uniq values: ", dataframe.select(colname).n_unique())
    print(f"[INFO] Unique values are: ", dataframe.select(colname).unique().to_series().to_list()[:up_to])


@app.cell
def _(CATEGORICALS, df):
    _COL = "pclass"
    print_null_and_uniques(_COL, df)

    # Verdict action
    CATEGORICALS.add(_COL)

    df.plot.bar(
        x=f"{_COL}:N", y="count()", color="survived"
    ).properties(width=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. survived

    This is our prediction target rather than an input feature. It has no missing values, so we can keep it unchanged in the semantic dataset and let the next notebook separate it from the predictors.

    **Verdict:** Keep it as the `target`. Any train/test splitting strategy or class-imbalance handling belongs downstream.
    """)
    return


@app.cell
def _(NUMERICS, df):
    _COL = "survived"
    print_null_and_uniques(_COL, df)

    # Verdict action
    NUMERICS.add(_COL)

    df.plot.bar(
        x=f"{_COL}:N", y="count()"
    ).properties(width=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. name

    Passenger names are far too high-cardinality to keep directly, but they do contain a compact semantic signal: honorific titles such as `Mr`, `Mrs`, `Miss`, or `Dr`.

    **Verdict:** Extract a deterministic `title` feature from `name` and drop the original `name` column. This is semantic parsing, not model preprocessing.
    """)
    return


@app.cell
def _(CATEGORICALS, DROP_COLS, df):
    _COL = "name"
    _NEW_TITLE = "title"
    print_null_and_uniques(_COL, df)

    df_name = df.with_columns(
        pl.col(_COL)
          .str.extract(r",\s*([A-Za-z]+)", 1)
          .fill_null("Unknown")
          .alias(_NEW_TITLE)
    )

    print_null_and_uniques(_NEW_TITLE, df_name)

    CATEGORICALS.add(_NEW_TITLE)
    DROP_COLS.add(_COL)

    df_name.select([_COL, _NEW_TITLE])
    return (df_name,)


@app.cell
def _(df_name):
    alt.Chart(df_name).mark_bar().encode(
        x=alt.X("title:N", sort="-y"),
        y=alt.Y("count():Q", scale=alt.Scale(type="log"))
    ).properties(width=900)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. sex

    This is already a small categorical column with clear meaning and no extra parsing required.

    **Verdict:** Keep it as-is as a categorical feature. Encoding is deferred to the modeling pipeline.
    """)
    return


@app.cell
def _(CATEGORICALS, df_name):
    _COL = "sex"
    print(f"[INFO] The {_COL} has null values: ", df_name.select(_COL).null_count().item())
    print(f"[INFO] Unique values are: ", df_name.select(_COL).unique().to_series().to_list())

    # Verdict action
    CATEGORICALS.add(_COL)

    df_name.plot.bar(
        x=f"{_COL}:N", y="count()", color="survived"
    ).properties(width=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. age

    `age` is already a meaningful numeric feature, and missing values here should remain visible to the downstream pipeline.

    **Verdict:** Keep it as-is. We do not impute missing ages in this notebook because imputation is fitted from data and belongs with the model pipeline.
    """)
    return


@app.cell
def _(df_name):
    _COL = "age"
    print(f"[INFO] The {_COL} has null values: ", df_name.select(_COL).null_count().item())
    print(f"[INFO] Unique values are: ", df_name.select(_COL).unique().to_series().to_list()[:30])

    df_name.plot.bar(
        x=f"{_COL}:Q", y="count()", color="survived"
    ).properties(width=900)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 & 7. sibsp and parch

    These two columns describe family relationships aboard the ship and are easier to reason about together than separately.

    We can combine them into a deterministic `family` feature using `sibsp + parch + 1`. The `+ 1` matters because the passenger themselves count as part of their travel group. That gives us a second deterministic feature, `is_alone`, defined as `family == 1`.

    **Verdict:** Derive `family` and `is_alone`, then drop `sibsp` and `parch` from the semantic output.
    """)
    return


@app.cell
def _(df_name):
    df_name.plot.bar(
        x=f"sibsp:N", y="count()", color="survived"
    ).properties(width=900)
    return


@app.cell
def _(DROP_COLS, NUMERICS, df_name):
    _COLa = "sibsp"
    _COLb = "parch"
    print(f"[INFO] The {_COLa} has null values: ", df_name.select(_COLa).null_count().item())
    print(f"[INFO] Unique values are: ", df_name.select(_COLa).unique().to_series().to_list())
    print(f"[INFO] The {_COLb} has null values: ", df_name.select(_COLb).null_count().item())
    print(f"[INFO] Unique values are: ", df_name.select(_COLb).unique().to_series().to_list())

    _NEW_FAMILY = "family"
    _NEW_ALONE = "is_alone"

    df_family = df_name.with_columns(
        (pl.col(_COLa) + pl.col(_COLb) + 1).alias(_NEW_FAMILY),
    )

    df_family = df_family.with_columns(
        (pl.col(_NEW_FAMILY) == 1).alias(_NEW_ALONE)
    )

    # Verdict
    DROP_COLS.add(_COLa)
    DROP_COLS.add(_COLb)
    NUMERICS.add(_NEW_FAMILY)

    df_family.plot.bar(
        x=f"{_NEW_FAMILY}:N", y="count()", color="survived"
    ).properties(width=900)
    return (df_family,)


@app.cell
def _(df_family):
    df_family.plot.bar(
        x=f"is_alone:N", y="count()", color="survived"
    ).properties(width=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. ticket

    Raw ticket values are too messy and high-cardinality to keep directly, but the leading letters can still carry semantic information.

    **VERDICT:** Extract `ticketprefix` as a compact categorical feature and drop the original `ticket`. This is still semantic extraction, not encoding. Any later grouping of rare prefixes should happen in the modeling notebook if needed.
    """)
    return


@app.cell
def _(CATEGORICALS, DROP_COLS, df_family):
    _COL = "ticket"
    _NEW_TICKET = "ticketprefix"

    print(f"[INFO] The {_COL} has null values: ", df_family.select(_COL).null_count().item())
    print(f"[INFO] Unique values are: ", df_family.select(_COL).unique().to_series().to_list()[:30])

    df_ticket = df_family.with_columns(
        pl.col(_COL)
          .str.extract(r"^\s*([A-Za-z])", 1)   # keep only first leading letter
          .str.to_uppercase()
          .fill_null("NUMBER")
          .alias(_NEW_TICKET)
    )

    print(f"[INFO] Null values in {_NEW_TICKET}: ", df_ticket.select(_NEW_TICKET).null_count().item())
    print(f"[INFO] Unique values remaining: ", df_ticket.select(_NEW_TICKET).n_unique())
    print(f"[INFO] Unique values are: ", df_ticket.select(_NEW_TICKET).unique().to_series().to_list())

    CATEGORICALS.add(_NEW_TICKET)
    DROP_COLS.add(_COL)

    df_ticket.plot.bar(
        x=alt.X(f"{_NEW_TICKET}:N", sort="-y"),
        y="count()",
        color="survived"
    ).properties(width=900)
    return (df_ticket,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. fare

    `fare` is already a meaningful numeric column, even if it is skewed and contains a few very large values.

    **VERDICT:** Keep it as-is. Binning, clipping, transforming, or scaling the distribution would be modeling decisions, so they are intentionally deferred.
    """)
    return


@app.cell
def _(NUMERICS, df_name, df_ticket):
    _COL = "fare"
    print(f"[INFO] The {_COL} has null values: ", df_name.select(_COL).null_count().item())
    print(f"[INFO] Unique values are: ", df_name.select(_COL).unique().to_series().to_list()[:30])

    # Verdict
    NUMERICS.add(_COL)

    df_ticket.plot.bar(
        x=f"{_COL}:Q", y="count()", color="survived"
    ).properties(width=900)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. cabin

    Full cabin identifiers are sparse and high-cardinality, but the leading cabin letter still carries semantic information about the deck.

    **Verdict:** Extract `cabinprefix` and drop the original `cabin` column. Rows without a cabin value are labeled `UNKNOWN` in this derived feature so that the semantic dataset preserves the fact that cabin information was absent.

    This is different from downstream imputation: we are not guessing a deck, we are creating an explicit category that means cabin information was missing in the source data.
    """)
    return


@app.cell
def _(CATEGORICALS, DROP_COLS, df_family, df_ticket):
    _COL = "cabin"
    _NEW_CABIN = "cabinprefix"

    print(f"[INFO] The {_COL} has null values: ", df_family.select(_COL).null_count().item())
    print(f"[INFO] Unique values are: ", df_family.select(_COL).unique().to_series().to_list()[:30])

    df_cabin = df_ticket.with_columns(
        pl.col(_COL)
          .str.extract(r"^\s*([A-Za-z0-9])", 1)   # keep only first leading letter or number
          .str.to_uppercase()
          .fill_null("UNKNOWN")
          .alias(_NEW_CABIN)
    )

    print(f"[INFO] Null values in {_NEW_CABIN}: ", df_cabin.select(_NEW_CABIN).null_count().item())
    print(f"[INFO] Unique values remaining: ", df_cabin.select(_NEW_CABIN).n_unique())
    print(f"[INFO] Unique values are: ", df_cabin.select(_NEW_CABIN).unique().to_series().to_list())

    # Verdict
    CATEGORICALS.add(_NEW_CABIN)
    DROP_COLS.add(_COL)

    df_cabin.plot.bar(
        x=alt.X(f"{_NEW_CABIN}:N", sort="-y"),
        y="count()",
        color="survived"
    ).properties(width=900)
    return (df_cabin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. embarked

    This is an interesting case. We have only two missing values. Instead of imputing, we can find that these individuals are [Rose Amélie Icard](https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html) and [Martha Evelyn Stone](https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html).

    You can visit the encyclopedia by clicking the links. The site states, e.g., that:

    > "Martha was awake in bed when the Titanic struck the iceberg. She slipped a kimono over her night dress, put on her slippers, and went out into the corridor and found other people similarly attired."
    >
    > – Encyclopedia Titanica

    **VERDICT:** Both individuals boarded the ship in **Southampton**, so we can fill in the value **S** and rename the semantic output column to `embark_at`.

    This is one of the few cases where filling a missing value is still acceptable in this notebook, because the value is resolved from external domain evidence rather than estimated from the dataset. If we did not have that evidence, the null should remain for the modeling pipeline to handle later.
    """)
    return


@app.cell
def _(df_cabin):
    df_cabin.filter(pl.col("embarked").is_null())
    return


@app.cell
def _(CATEGORICALS, DROP_COLS, df_cabin):
    _COL = "embarked"
    _NEW_EMBARKED = "embark_at"

    print(f"[INFO] The {_COL} has null values: ", df_cabin.select(_COL).null_count().item())
    print(f"[INFO] Unique values are: ", df_cabin.select(_COL).unique().to_series().to_list()[:30])

    # Verdict
    DROP_COLS.add(_COL)
    CATEGORICALS.add(_NEW_EMBARKED)

    df_embarked = df_cabin.with_columns(
        pl.col(_COL).fill_null("S").alias(_NEW_EMBARKED),
    )

    df_embarked.plot.bar(
        x=f"{_NEW_EMBARKED}:N", y="count()", color="survived"
    ).properties(width=900)
    return (df_embarked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Embark and Ticket sanity check

    We now have two features that may both encode some location information: ticket prefix and embarkation port. Before moving on, it is worth checking that they are not just duplicates of each other.

    The letters in the embarkation column mean:

    * C = Cherbourg (France)
    * Q = Queenstown (Ireland) - now known as Cobh
    * S = Southampton (UK)
    """)
    return


@app.cell
def _(df_embarked):
    _plot_counts = (
        df_embarked
        .select([
            pl.col("embark_at"),
            pl.col("ticketprefix"),
        ])
        .group_by(["embark_at", "ticketprefix"])
        .len()
    )

    _heatmap = (
        alt.Chart(_plot_counts)
        .mark_rect()
        .encode(
            x=alt.X("ticketprefix:N", title="Ticket prefix"),
            y=alt.Y("embark_at:N", title="Embarked at"),
            color=alt.Color("len:Q", title="Count", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("embark_at:N", title="Embarked at"),
                alt.Tooltip("ticketprefix:N", title="Ticket prefix"),
                alt.Tooltip("len:Q", title="Count"),
            ],
        )
        .properties(
            width=700,
            height=250,
        )
    )

    _heatmap
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. boat

    The `boat` indicates that you made it to the life boat. If you don't make it to the life boat, your chances of survival in the North Atlantic Ocean are fairly low. If you do your research and read about the survivors of Titanic, you will learn that about 40 of the 1500 who were left in the Titanic after the last lifeboat launched managed to survive. One of them was [John "Jack" Thayer](https://www.encyclopedia-titanica.org/titanic-survivor/john-borland-thayer-jr.html). This is not the same Jack as in the Cameron film, but ended up in the water in a similar manner:

    > "A sort while later Jack jumped out, feet first. He surfaced well clear of the ship, he felt he was pushed away from the ship by some force."
    >
    > – Jack Thayer

    Wonder if he is in the dataset?

    **Verdict:** Drop. `boat` leaks post-incident information that would not be available at prediction time.
    """)
    return


@app.cell
def _(DROP_COLS, df_embarked):
    # Verdict
    DROP_COLS.add("boat")

    (
        df_embarked
            .filter(pl.col("name").str.contains("Thayer"))
            .filter(pl.col("age").eq(17))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. body

    If you already have a body bag number, there isn't a lot to predict.

    **Verdict:** Drop. This is an even clearer leakage column than `boat`.
    """)
    return


@app.cell
def _(DROP_COLS):
    DROP_COLS.add("body")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. home.dest

    The `home.dest` contains various information of their either home town or country or destination. It *might* be somehow useful, assuming that non-native English speakers wouldn't have access to all safety instructions due to language barrier. Let's investigate the Finns aboard on the ship.

    **Verdict:** Drop. There may be signal here, but turning this free-text field into a useful semantic feature would require a much larger mapping project than this introductory notebook should take on.
    """)
    return


@app.cell
def _(DROP_COLS, df_embarked):
    # Verdict
    DROP_COLS.add("home.dest")

    df_embarked.filter(pl.col("home.dest").str.to_lowercase().str.contains("finla"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Write to disk

    We are now ready to write the semantic dataset to disk.

    The Parquet file will contain only the columns we decided to keep after deterministic feature extraction and leakage removal. The sidecar JSON file records which columns are categorical, numeric, dropped, and which column is the target so that the next notebook can build a model pipeline without repeating the EDA work.
    """)
    return


@app.cell
def _(CATEGORICALS, DROP_COLS, NUMERICS, TARGET, df_embarked):
    metadata = {
        "categoricals": sorted(CATEGORICALS),
        "numerics": sorted(NUMERICS),
        "drop_cols": sorted(DROP_COLS),
        "target": list(TARGET),
    }
    print(metadata)

    df_write = df_embarked.drop(DROP_COLS)

    # View
    df_write
    return df_write, metadata


@app.cell
def _(TITANIC_META_FILE, TITANIC_OUT_FILE, df_write, metadata):
    # Write Parquet data
    df_write.write_parquet(TITANIC_OUT_FILE)

    # Write sidecar metadata
    with open(TITANIC_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote data to: {TITANIC_OUT_FILE}")
    print(f"[INFO] Wrote metadata to: {TITANIC_META_FILE}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion

    This notebook produced a semantic Titanic dataset intended to be consumed by the next notebook.

    What we kept:

    * categorical inputs such as `pclass` and `sex`
    * numeric inputs such as `age` and `fare`
    * the target column `survived`

    What we derived deterministically:

    * `title` from `name`
    * `family` from `sibsp + parch + 1`
    * `is_alone` from `family == 1`
    * `ticketprefix` from the leading ticket letters
    * `cabinprefix` from the leading cabin letter, with `UNKNOWN` marking missing cabin information
    * `embark_at` from `embarked`, including a deterministic fix for the two missing values

    What we dropped:

    * obvious leakage columns such as `boat` and `body`
    * high-cardinality raw columns that were replaced by semantic features such as `name`, `ticket`, `cabin`, `sibsp`, `parch`, and `embarked`
    * `home.dest`, because converting it into a robust semantic feature would require a larger project than this notebook should own

    What we deliberately did **not** do:

    * impute unresolved missing values such as `age`
    * encode categorical columns
    * scale numeric columns
    * make frequency-based or model-dependent feature decisions

    That work belongs in the next notebook, which should stay focused on loading the semantic Parquet, splitting `X` and `y`, defining the estimator pipeline, and evaluating the model.

    One cleanup item remains for the future: the output feature names are currently useful but not fully standardized. For now, this notebook documents the existing contract exactly as written so that downstream code can rely on it.
    """)
    return


if __name__ == "__main__":
    app.run()
