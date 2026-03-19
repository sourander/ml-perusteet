import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt
    import polars.selectors as cs


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

    In this document, we will figure out how to handle each column. Also, this will work as a test that your environment works as expected. To start, you need to know how to unzip files. Binary files are data files are on Git LFS directory, and you must uncompress them to `data/` directory. You can do this like so...

    ```bash
    cd notebook
    unzip gitlfs-store/titanic.zip
    ```

    ## Goal of this file

    We want to produce a `titanic_semantic.parquet` file, which has only **deterministic, semantic feature extraction** applied into this. For example, we can produce field like `is_alone` as `family_size == 1` or parse the persons title, like `Dr. (doctor)`, from name. We can also completely drop some columns or decide to rename them. Also, all values that should be `NULL`, like string values (`["\", " ", "N/A", "None that I know :)"]`) will be turned to actual `NULL` values.

    Anything else must be left for the next script to handle These include things that are:

    * fitted from training data (leakage risk if done globally)
    * model-dependent
    * a hyperparameter we want to tune
    * imputing the NULL values
    """)
    return


@app.cell
def _():
    TITANIC_FILE = "data/titanic/titanic.csv"
    TITANIC_OUT_FILE = "data/titanic/titanic.parquet"

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
    return TITANIC_FILE, TITANIC_NULL_VALUES, TITANIC_OUT_FILE, TITANIC_SCHEMA


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

    A simple CSV does not contain any *schema information*, roughly meaning the data types of each columns. Only information that can be read from the file is what is the name of the field (if header exists) and whether the field contains a number, a string or a missing value.

    For example, it makes sense to read values that can only be `0/1` as booleans. This does not simply just save our RAM (by storing less bits per value), but also makes it easier to visualize and process the data.

    The data has been originally downloaded from ~~`http://biostat.mc.vanderbilt.edu/DataSets`~~  and that site is no longer online. You can find the metadata at [TitanicMETA.pdf](http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf), but the table description is replicated below as Markdown table for your convenience.


    | Field      | Description                                                             |
    |------------|-------------------------------------------------------------------------|
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
    # 30,000 feet view

    The whole dataset can be checked at once. There are multiple ways to do this, including also...

    ```python
    from skrub import TableReport
    TableReport(df)
    ```

    The code block above would run nearly 1 minute and print an interactive raport you can investigate. We will simply check a typical **scatter matrix**. Nominal (our string) fields are hard to plot, so we will keep only numerical fields.
    """)
    return


@app.cell
def _(df):
    # Keep all numeric columns except the survived
    _numeric_cols = list(df.select(cs.numeric()).columns)
    _numeric_cols.remove("survived")

    # Look into this too!
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

    We will be going through these 14 columns. As we go, we will categorize each one into one of the bins (e.g. `CATEGORICALS` for low-cardinality categoricals).
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

    **Verdict:** This is clearly a low cardinality column.
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

    **Verdict:** This is our `target`. It has no missing values, so we can just keep it as it is. It is mildly imbalanced; we may want to take this into account when splitting into training and testing sets.
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

    **Verdict:** Name is too high cardinality, but it contains titles, which might be useful. Let's create a new column `title` and drop the old `name`
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

    **Verdict:** We can keep this as it is.
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

    **Verdict:** We can leave it as it is.
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

    **VERDICT:** These two columns have a lot in common. We can create a new column for family size. Let's also create another one to be a boolean column indicating if one is alone.
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

    **VERDICT:** The letters in the ticket may contain important information. It seems that e.g. `S` points to Southhampton.
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

    **VERDICT:** Keep as it is. We might want to either bin this, or cap to e.g. 300 to reduce the effect of the 4 outliers (with a fare of 500+).
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

    **Verdict:** Let's do the same trick as with ticket. Keep the starting letter (or number, which don't exist). We will simply mark all others as UNKNOWN. These is such as high volume of these that imputation strategies would not end up being useful.
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

    **VERDICT:** Both individuals boarded the ship in **Southampton**. Thus, we can fill in the value **S**.
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

    We have two fields now containing city-knowledge. The ticket prefix `S...` and embark_at `S` both relate to Southhampton, most likely. Let's make sure that this is not quite 1:1 duplicate information.

    The letters in the embarkment column should mean:

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

    **Verdict:** You would not know whether you make it into a life boat or not when before that has happened. This column needs to be dropped.
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

    **Verdict:** Drop.
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

    **Verdict:** It surely is possible to use this field to infer information about where they are from, but this would require a fair amount of mapping (or utilizing a simple BERT language model of sorts). Thus, let's drop it.
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

    We are now ready to write the dataset to disk as a Parquet file (or any other chosen file format).
    """)
    return


@app.cell
def _(CATEGORICALS, DROP_COLS, NUMERICS, TARGET):
    print(CATEGORICALS) # {'cabinprefix', 'embark_at', 'pclass', 'sex', 'ticketprefix', 'title'}
    print(NUMERICS)     # {'fare', 'survived', 'family'}
    print(DROP_COLS)    # {'cabin', 'parch', 'home.dest', 'body', 'name', 'ticket', 'embarked', 'sibsp'}
    print(TARGET)       # ['survived']
    return


@app.cell
def _(DROP_COLS, df_embarked):
    df_write = df_embarked.drop(DROP_COLS)

    # View
    df_write
    return (df_write,)


@app.cell
def _(TITANIC_OUT_FILE, df_write):
    # Write to disk
    df_write.write_parquet(TITANIC_OUT_FILE)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion

    TODO: Summarize here what was done to the dataset.
    """)
    return


if __name__ == "__main__":
    app.run()
