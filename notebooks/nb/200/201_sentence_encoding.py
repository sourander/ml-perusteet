import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    from scipy.stats import beta


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sentence Encoding (goal: TF-IDF)

    Machine learning models cannot process raw text directly; they require numerical representations. **TF-IDF** is a foundational technique in Natural Language Processing (NLP) used to convert text into meaningful numbers. It scores the importance of a word (a "token") in a document based on a larger collection of documents (a "corpus").

    **The Mathematical Components:**
    * **TF(d, t): Term Frequency** = The number of times token $t$ appears in document $d$.
    * **DF(t): Document Frequency** = The number of documents that contain the token $t$.
    * **N**: The count of documents in the dataset.
    * **IDF(t): Inverse Document Frequency** = $\log(N / DF(t))$, where $N$ is the total number of documents. The logarithm helps dampen the effect of very large numbers.
    * **TF-IDF(d, t)** = $TF(d, t) \times IDF(t)$


    ## Our Dataset
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
        },
    )

    # Let's add an index column so that we can later on join the tokens back
    df = df.with_row_index("doc_id")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tokenizing

    Tokenization is the process of breaking text down into its smallest meaningful parts (tokens). For a computer to analyze sentences, we must first break them into words.

    We achieve this by extracting letters into a list and forcing the strings to lowercase, so "Hyvä" and "hyvä" are treated equally.
    """)
    return


@app.cell
def _(df):
    df_token = df.with_columns(
        tokens=pl.col("feedback").str.to_lowercase().str.extract_all(r"\p{L}+")
    )

    df_token
    return (df_token,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And then, explore this list-type column into rows. Essentially, we go from `nested wide` to `tall` table format.
    """)
    return


@app.cell
def _(df_token):
    df_tokens = (
        df_token.select(["doc_id", "tokens"])
        .explode("tokens")
        .rename({"tokens": "token"})
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
    )
    df_tokens
    return (df_tokens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tokens and Term Frequency (TF)

    First, let's normalize the term frequency per document. It is important to count how often a token appears in a document, but a long text naturally has larger counts than a short one. By dividing the term count by the document length, we create a normalized Term Frequency ($TF$).

    Note that you could also derive this from the previously exploded table by doing a GROUP BY, like...

    ```python
    doc_lengths = df_tokens.group_by("doc_id").len().rename({"len": "doc_len"})
    ```
    """)
    return


@app.cell
def _(df_token):
    doc_lengths = df_token.with_columns(doc_len=pl.col("tokens").list.len()).select("doc_id", "doc_len")
    doc_lengths
    return (doc_lengths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And then...
    """)
    return


@app.cell
def _(df_tokens, doc_lengths):
    TF = (
        df_tokens.group_by(["doc_id", "token"])
        .len()
        .rename({"len": "term_count"})
        .join(doc_lengths, on="doc_id", how="left")
        .with_columns((pl.col("term_count") / pl.col("doc_len")).alias("tf"))
    )
    TF
    return (TF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Document Frequency (DF)

    Next, we count how many unique documents (in our case, feedback entries) contain a particular token.
    """)
    return


@app.cell
def _(df_tokens):
    DF = (
        df_tokens.select(["doc_id", "token"])
        .unique()
        .group_by("token")
        .len()
        .rename({"len": "df"})
    )

    DF
    return (DF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inverse Document Frequency (IDF)

    Here we penalize common words. Why? Because a token (like "selkeä" or "harjoitus") that appears in nearly all feedback texts loses its predictive value. By measuring the $IDF$, frequent terms are assigned a smaller weight and rare but specific words get a boost.

    Note that the formula used below is the **non-smoothed** version:
    $$IDF(t) = \log\left(\frac{N}{DF(t)}\right)$$

    In practice, libraries (e.g., scikit-learn) use a smoothed IDF. This adds a small correction to the formula so that the weighting behaves more robustly and terms that appear in every training document do not collapse to a zero-only effect.

    $$IDF(t) = \log\left(\frac{N + 1}{DF(t) + 1}\right) + 1$$
    """)
    return


@app.cell
def _(DF, df):
    N_DOCS = df.height

    IDF = DF.with_columns((pl.lit(N_DOCS) / pl.col("df")).log().alias("idf"))

    IDF
    return (IDF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Term Frequency-Inverse Document Frequency

    Ah, we finally arrived to the TF-IDF destination.
    """)
    return


@app.cell
def _(IDF, TF):
    TF_IDF = (
        TF.join(IDF.select(["token", "idf"]), on="token", how="left")
        .with_columns(
            (pl.col("tf") * pl.col("idf")).alias("tf_idf")
        )
        .sort(["doc_id", "tf_idf"], descending=[False, True])
    )

    TF_IDF
    return (TF_IDF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Join with the Original DataFrame
    """)
    return


@app.cell
def _(TF_IDF, df):
    TF_IDF_DOCS = (
        TF_IDF.join(
            df.select(["doc_id", "feedback", "age", "level", "lang", "passed"]),
            on="doc_id",
            how="left",
        )
        .select(
            [
                "doc_id",
                "feedback",
                "token",
                "term_count",
                "idf",
                "tf_idf",
                "age",
                "level",
                "lang",
                "passed",
            ]
        )
        .sort(["doc_id", "tf_idf"], descending=[False, True])
    )

    TF_IDF_DOCS
    return (TF_IDF_DOCS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Wide representations

    Now that we have computed our TF-IDF values for the text tokens, we need to restructure it so that it resembles a typical data frame used for Machine Learning in Pandas and Scikit-Learn: columns contain properties/variables, rows contain instances. One row is one document.

    ## Our Vocabulary
    """)
    return


@app.cell
def _(TF_IDF_DOCS):
    vocabulary = (
        TF_IDF_DOCS.select("token").unique().sort("token").get_column("token").to_list()
    )
    vocabulary
    return (vocabulary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Original + TF-IDF vectorized columns

    We pivot the tokens into wide format, turning each token into its own feature (column). The numerical value associated with the token is its TF-IDF score.
    """)
    return


@app.cell
def _(TF_IDF_DOCS, df, vocabulary):
    df_with_tf_idf = (
        TF_IDF_DOCS.pivot(
            values="tf_idf",
            index=df.columns,
            on="token",
        )
        .with_columns(pl.col(vocabulary).fill_null(0.0))
        .select(df.columns + vocabulary)
        .sort("doc_id")
    )

    df_with_tf_idf
    return (df_with_tf_idf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Original + Count Vectorized columns (BoW)

    This is the Bag of Words from the lesson material.
    """)
    return


@app.cell
def _(TF_IDF_DOCS, df, vocabulary):
    df_with_countvectorized = (
        TF_IDF_DOCS.pivot(
            values="term_count",
            index=df.columns,
            on="token",
        )
        .with_columns(pl.col(vocabulary).fill_null(0))
        .select(df.columns + vocabulary)
        .sort("doc_id")
    )

    df_with_countvectorized
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: Normalize

    The scikit-learn's implementation normalizes the features so that their sum equals 1. The default normalization is L2. We can implement that using a function that has been copy-pasted from Polars-DS Extension's site – and simplified a bit. See the original docs at [polars-ds: Numerical Functions](https://polars-ds-extension.readthedocs.io/en/latest/num.html)
    """)
    return


@app.cell
def _(df_with_tf_idf, vocabulary):
    def str_to_expr(x: str | pl.Expr) -> pl.Expr:
        """
        Copyright note! This function is based on polars-ds _utils.py file:
        https://github.com/abstractqqq/polars_ds_extension/blob/main/python/polars_ds/_utils.py
        Currently, the matching function is named to_expr.
        """
        return pl.col(x) if isinstance(x, str) else x


    def l2_sq_horizontal(*v: str | pl.Expr) -> pl.Expr:
        """
        Copyright note! This function is based on polars-ds num.py file:
        https://github.com/abstractqqq/polars_ds_extension/blob/main/python/polars_ds/exprs/num.py
        """
        exprs = list(v)
        return pl.sum_horizontal(str_to_expr(x).pow(2) for x in exprs) / len(exprs)


    df_with_tf_idf_l2 = (
        df_with_tf_idf
        .with_columns(
            magnitude=l2_sq_horizontal(*vocabulary).sqrt()
        )
        .with_columns(
            pl.col(vocabulary)
            .truediv(pl.col("magnitude"))
            .fill_nan(0.0)
            .fill_null(0.0)
        )
        # .drop("magnitude") - naturally, you wouldn't need this. We keep it for reference.
    )

    df_with_tf_idf_l2
    return


if __name__ == "__main__":
    app.run()
