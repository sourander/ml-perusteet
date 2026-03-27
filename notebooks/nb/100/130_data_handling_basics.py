import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import polars as pl
    import altair as alt


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Handling Basics

    This Notebook is reserved for introducing the data loading and processing libraries that are heavily used in this course. You are expected to utilize these libraries instead of e.g. Python loops. In current version of the course, these are...

    * **NumPy**: vector and matrix operations (especially in *from Scratch* implementations)
    * **Polars**: file input and output, tabular Data Frames, aggregations
    * **Altair**: plotting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pandas

    Let's look at how one would perform a typical linear algebra operation without/with NumPy.

    ## Matrix without NumPy

    Let's perform a simple linear algebra operation using Python-only solution. The operation we will perform is a dot product, like..

    $$
    \hat{y} = Xw
    $$

    Where the matrix X is...

    $$
    \begin{bmatrix}
    \hat{y}_{1} \\
    \hat{y}_{2} \\
    \hat{y}_{3} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    x_{1,1} & x_{1,2} \\
    x_{2,1} & x_{2,2} \\
    x_{3,1} & x_{3,2} \\
    \end{bmatrix}
    \begin{bmatrix}
    w_{1} \\
    w_{2} \\
    \end{bmatrix}
    $$

    And the operation is performed as...


    $$
    \begin{bmatrix}
    \hat{y}_{1} \\
    \hat{y}_{2} \\
    \hat{y}_{3} \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    11 & 12 \\
    21 & 22 \\
    31 & 32 \\
    \end{bmatrix}
    \begin{bmatrix}
    1 \\
    2 \\
    \end{bmatrix}
    =
    \begin{bmatrix}
    11 \cdot 1 + 12 \cdot 2 \\
    21 \cdot 1 + 22 \cdot 2 \\
    31 \cdot 1 + 32 \cdot 2 \\
    \end{bmatrix}
    $$
    """)
    return


@app.cell
def _():
    def vector_dot(u, v):
        return sum(u_i * v_i for u_i, v_i in zip(u, v))

    def matrix_dot(X, w):
        return [vector_dot(x, w) for x in X]

    # Matrix representation of a dataset with 3 rows, 2 features
    X_py = [
        (11, 22),
        (21, 22),
        (31, 22),
    ]

    # Vector (row-direction) vector representation of some parameters/coefficients
    w_py = (1, 2)

    y_hat_py = matrix_dot(X_py, w_py)
    print(y_hat_py)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix with NumPy
    """)
    return


@app.cell
def _():
    X = np.array(
    [
        (11, 12),
        (21, 22),
        (31, 32),
    ]
    )

    w = np.array([1, 2])

    # Option 1: Using the dot method
    y_hat = X.dot(w)

    # Option 2: Using the matmul operator
    y_hat = X @ w
    print(y_hat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polars

    Pandas is the mother of all data analytics in Python. However, it does have it's quircks. During this course, we will be using a modern alternative to Pandas called Polars. Since Pandas is still very popular, it is assumed that you have at least seen the Pandas syntax in other courses. Thus, this quick guide will be a **migration guide**. This is based on Polars' docs' [Coming from Pandas](https://docs.pola.rs/user-guide/migration/pandas/) guide, the [Getting started](https://docs.pola.rs/user-guide/getting-started/) guide, and the Chapter 3, *Moving from pandas to Polars*, of a book called **Python Polars: The Definitive Guide** by Jeroen Janssens and Thijs Nieuwdorp (O'Reilly 2025).

    Polars is, by convention, imported with an alias `pl`.

    ## 1. Polars in Columnar

    Polars stores data in a columnar format, unlike Pandas, which uses row‑based blocks. Columnar execution gives Polars much better performance in analytics tasks.
    """)
    return


@app.cell
def _():
    data = {
        "name": ["Alice", "Bob", "Charlie", "Jack", "Rose", "Joanna"],
        "age": [24, 30, 29, 24, 23, 42],
        "height_cm": [165, 175, 168, 178, 175, 182],
    }

    df_poorly_typed = pl.DataFrame(data)
    df_poorly_typed
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Polars is Strict

    Polars requires consistent column types. Setting dtypes explicitly helps avoid all sorts of problems and also makes processing more memory efficient. As a data professional, *inferring schema* automatically should be something you usually avoid.
    """)
    return


@app.cell
def _(data):
    df = pl.DataFrame(
        data,
        schema={
            "name": pl.Utf8,
            "age": pl.Int8,
            "height_cm": pl.Int16,
        }
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Index, Index, Index, Index, ...

    Polars intentionally does not support multi‑index or arbitrary indices like `RangeIndex`. A Polars DataFrame behaves like a SQL table: if you need an index, add one as a column
    """)
    return


@app.cell
def _(data):
    n = len(next(iter(data.values())))
    index = {"id": list(range(n))}
    data_indexed = data | index 

    df_with_index = pl.DataFrame(
        data_indexed,
        schema={
            "id": pl.Int64,
            "name": pl.Utf8,
            "age": pl.Int8,
            "height_cm": pl.Int16,
        }
    )
    df_with_index
    return index, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Slicing 'n' Dicing

    The Pandas DataFrames have row and column indicies and they can be sliced using the `pandas_df[4:2]` and `.iloc[]` and `loc[]` appoaches.

    **Unlearn this**. Use `select` and `filter`. This is a bit like how it's done in SQL too.
    """)
    return


@app.cell
def _(df):
    df.select(["name", "age"])
    return


@app.cell
def _(df):
    df.filter(pl.col("age") > 25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.Assume new object

    Mutating objects can lead to hard-to-solve problems, especially with Marimo, which creates a computational dependency DAG for cells. We don't do that here.

    > "With pandas, every operation is eager and many happen in place. Eager means that every operation is executed immediately. In place means that the operation does not return a new DataFrame, but modifies the existing one. [...] With Polars, none of the operations are in place by default"
    >
    > — Janssens & Nieuwdorp

    Note that this does not mean that memory is being used carelessly. Polars does not copy existing data when doing e.g. `df_B = df_A.with_columns(...)`; it produces a new DataFrame object referencing the same memory where possible (zero‑copy).
    """)
    return


@app.cell
def _(df, index, n):
    spam = ["spam"] * n


    df.with_columns(
        id=pl.from_dict(index).to_series(),
        spam=pl.Series("spam", spam),
        age=pl.col("age") > 30,
        odd_height=(pl.col("height_cm") & 1 == 0),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Expressions

    Polars expressions feel similar to SQL operations. Below are a few to demonstrate.

    ### GROUP BY
    """)
    return


@app.cell
def _():
    df_people = pl.DataFrame({
        "team": ["A", "A", "B", "B", "B"],
        "name": ["Alice", "Bob", "Charlie", "Dora", "Evan"],
        "score": [5, 7, 8, 6, 9]
    })


    df_people.group_by("team").agg(
        avg_score = pl.col("score").mean(),
        max_score = pl.col("score").max()
    )
    return (df_people,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### OVER PARTITION BY
    """)
    return


@app.cell
def _(df_people):
    df_people.with_columns(
        team_avg = pl.col("score").mean().over("team")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### SELECT EXCEPT
    """)
    return


@app.cell
def _(df_people):
    df_people.select(pl.exclude("team"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. UDFs

    If something cannot be expressed with Polars operations, you can use a Python UDF via map_batches().
    Here we transform data in Python while preserving DataFrame chunking. This is slow and should be avoided if expressions provide the same thing.
    """)
    return


@app.cell
def _(df_people):
    # Slow Python UDF — intentionally not vectorized
    def custom_python_function(series: pl.Series) -> pl.Series:
        # Perform character‑by‑character uppercasing in pure Python
        values = []
        for s in series:
            if s is None:
                values.append(None)
            else:
                # intentionally slow
                values.append("".join([ch.upper() for ch in s]))
        return pl.Series(values)

    df_people.with_columns(
        pl.col("name")
          .map_batches(custom_python_function, return_dtype=pl.String)
          .alias("name_upper_slow")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of the dum dum example above, you should be utilizing Polars' Expressions.
    """)
    return


@app.cell
def _(df_people):
    df_people.with_columns(
        name_upper_fast=pl.col("name").str.to_uppercase()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. (Optional) Brief Note About Lazy Mode

    Polars also supports Lazy mode, where operations are not executed immediately.
    Instead, Polars builds a query plan and optimizes it before execution (similar to how SQL engines and tools like Apache Spark optimize queries). This also relates to the `map_batches()` mentioned above. In (default) eager mode, `map_batches()` receives the whole column as a single Series, i.e. `batch == whole Series`.

    We will not use Lazy mode in this course, but it is a great way to learn how modern query optimizers work in systems such as Apache Spark (`pyspark`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Altair

    Altair uses the Vega‑Lite grammar and is fully declarative: you describe what you want, not how to draw it. This is similar to SQL. Thus, the shift from e.g. Matplotlib ---> Altair is similar as Pandas ---> Polars.

    Altair produces highly interactive diagrams, which fits our Marimo Notebooks like a glove.

    **NOTE ABOUT AI:** Remember that this is a course about Machine Learning, not about plotting. Utilizing language models for producing the Altair code is fully acceptable. What you need to do is **interpret** and **understand** and **explain** what the resulting diagrams represents.

    ## 1. Declarative

    Note that we simply describe what goes where. Altair does the rest. The syntax `name:N` means that the column is NOMINAL. The `Q` is QUANTITATIVE.
    """)
    return


@app.cell
def _(df_people):
    alt.Chart(df_people).mark_bar().encode(
        x="name:N",
        y="score:Q"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Encodings

    In Altair, you don’t manually loop over data or instruct “draw this bar here.” You don't handle `ax` and `fig`. Instead, you assign encodings, which map data columns onto visual channels (position, color, size, shape, etc).
    """)
    return


@app.cell
def _(df):
    alt.Chart(df).mark_circle(size=200).encode(
        x="age",
        y="height_cm",
        color="name"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Built-In Aggregation

    Often, you do not need to aggregate the data using Polars before plotting it. Altair can do it for you.
    """)
    return


@app.cell
def _(df_people):
    alt.Chart(df_people).mark_bar().encode(
        x="team",
        y="mean(score)"
    ).properties(width=400) # try also width="container"

    # Do also try:
    #   x=alt.X("title:N", sort="-y")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Layering of Charts

    Adding different kind of vertical lines, horizontal lines, heatmap matrix cell numbers and similar can be done by simple `adding up` two visualizations.
    """)
    return


@app.cell
def _(df):
    scatter = alt.Chart(df).mark_circle(size=150).encode(
        x="age",
        y="height_cm",
        color="name"
    )

    avg_rule = alt.Chart(df).mark_rule(color="red").encode(
        y="mean(height_cm)"
    )

    scatter + avg_rule
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Tooltips

    Altair makes interactive tooltips extremely simple. You only add a `tooltip=[...]` encoding that lists which fields should appear when the user hovers over a bar, point, or any other mark.
    """)
    return


@app.cell
def _(df_people):
    alt.Chart(df_people).mark_arc(innerRadius=50).encode(
        theta="count(team)",
        color="team:N",
        tooltip=["team", "count(team)"]
    )
    return


if __name__ == "__main__":
    app.run()
