import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    import math
    import json
    import io
    import re
    import polars as pl
    import altair as alt
    import seaborn as sns
    import matplotlib.pyplot as plt

    from pathlib import Path

    # This dataset has ~12k rows after reshaping, above Altair's default 5000-row embed limit
    alt.data_transformers.disable_max_rows()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # First EDA on the ySKILLS Longitudinal Dataset

    The original source of the data is Data in Brief Vol. 54: [Digital skills among youth: A dataset from a three-wave longitudinal survey in six European countries](https://www.sciencedirect.com/science/article/pii/S2352340924003652). The raw CSV is around 33 MB and contains 800+ columns, since the same questionnaire was repeated in three waves (`W1_`, `W2_`, `W3_` column prefixes).

    Our chosen business problem is: **Can we predict the `RISK101` variable using the other variables in the dataset?**

    `RISK101` is defined as experience with cyberhate in the past year: *"On the internet, you may encounter content that attacks certain groups or individuals (e.g., because of their skin colour, religion, nationality, gender, or sexuality)."*

    ## Goal of this file

    We want to produce a semantic Parquet artifact at `data/y_skills/y_skills_model_independent_transformations.parquet`. As in the Titanic EDA notebook, **semantic** here means: transforms that are deterministic, justified from the meaning of the raw column, and do not require fitting on training data. Another word for these would be *model-independent transformations*.

    This notebook is responsible for:

    * merging the three waves into a single tall (long-format) table (a risky move!)
    * inspecting columns and choosing what to keep (verdicts)
    * fixing the comma-decimal string encoding used in the raw CSV
    * standardizing all missingness (blanks and negative "not asked"/"don't know" codes) to real nulls
    * cherry-picking the derived columns relevant to our research question
    * saving a semantic Parquet plus sidecar metadata

    Anything that is fit (=learned) from data, depends on the downstream estimator, or acts like a tunable hyperparameter must be left for the next notebook. That includes imputing unresolved missing values, one-hot/ordinal encoding, and scaling numeric features.

    ## Optional exercise

    The raw CSV is not stored in this repository. If you want to redo this preparation yourself (e.g. to try different feature choices), download the dataset from the link above and place it at `data/y_skills/ySKILLS_longitudinal_dataset.csv`. Otherwise, the prepared Parquet file is provided for you via the course's Git LFS store.
    """)
    return


@app.cell
def _():
    YSKILLS_FILE = Path("data/y_skills/ySKILLS_longitudinal_dataset.csv")
    YSKILLS_OUT_FILE = Path("data/y_skills/y_skills_model_independent_transformations.parquet")
    YSKILLS_META_FILE = YSKILLS_OUT_FILE.with_suffix(".meta.json")

    assert(YSKILLS_FILE.exists()), "Read the comment above! You need to download this data if you want to run this Notebook."
    return YSKILLS_FILE, YSKILLS_META_FILE, YSKILLS_OUT_FILE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading a Latin-1, semicolon-delimited CSV

    The raw file uses `;` as a separator and is encoded as Latin-1 rather than UTF-8 (it contains accented characters from several European languages). Polars' `read_csv` only understands UTF-8 natively, so we decode the raw bytes as Latin-1 and re-encode them as UTF-8 before handing them to Polars.
    """)
    return


@app.cell
def _(YSKILLS_FILE):
    _raw_bytes = YSKILLS_FILE.read_bytes()
    _utf8_bytes = _raw_bytes.decode("latin-1").encode("utf-8")

    df_raw = pl.read_csv(
        io.BytesIO(_utf8_bytes),
        separator=";",
        null_values=["", " "],
        infer_schema_length=None,
    )

    df_raw
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Merge waves

    Many of the columns exist three times, since the same questionnaire has been performed three times: waves 1, 2 and 3. Each column is named with the wave number as a prefix, e.g. `W1_something`, `W2_something`, `W3_something`.

    Not every column follows this pattern (a handful of administrative columns such as `country` and `waves` exist only once), and not every identifier exists in all three waves. We find the identifiers that exist in all three waves, since those are the ones we can stack into a single tall table.
    """)
    return


@app.cell
def _(df_raw):
    _pattern = re.compile(r"^W([1-3])_(.+)$")

    _column_waves: dict[str, set[str]] = {}
    _non_matching_columns = []

    for _col in df_raw.columns:
        _match = _pattern.match(_col)
        if _match is None:
            _non_matching_columns.append(_col)
            continue
        _wave, _identifier = _match.groups()
        _column_waves.setdefault(_identifier, set()).add(_wave)

    OMNI_WAVE_COLUMNS = sorted(
        _identifier for _identifier, _waves in _column_waves.items() if len(_waves) == 3
    )
    _missing_wave_columns = sorted(
        _identifier for _identifier, _waves in _column_waves.items() if len(_waves) != 3
    )

    print("[INFO] Columns without a W1_/W2_/W3_ prefix:", _non_matching_columns)
    print("[INFO] Identifiers missing from at least one wave:", _missing_wave_columns)
    print("[INFO] Identifiers present in all 3 waves:", len(OMNI_WAVE_COLUMNS))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cherry-pick derived columns

    The full dataset has 800+ raw survey items. Instead of using them directly, we use the derived scale/proportion columns that the ySKILLS data dictionary already computes for us (e.g. `friends` is the mean of `FRIEND1a`, `FRIEND1b`, `FRIEND1c`).

    Some derived columns overlap with each other and would introduce multicollinearity, so we drop the redundant one of each pair, per the data dictionary:

    * `civic_dich` is a dichotomized duplicate of `civic` — dropped.
    * `skill_inf_pro`, `skill_comm_pro`, `skill_cont_pro`, `skill_overall_pro` are covered by the more complete `lit_inf_pro`, `lit_comm_pro`, `lit_cont_pro`, `lit_overall_pro` — dropped.

    We pick these columns before reshaping (rather than after) so that the wide-to-tall stacking below only has to reconcile the dtypes of the columns we actually need, instead of all 286 columns that happen to exist in all 3 waves.
    """)
    return


@app.cell
def _():
    BASIC_INFO_COLUMNS = ["country", "Age_year", "GENDER"]

    DERIVED_COLUMNS = [
        "friends",
        "family",
        "civic",
        "daily_activities",
        "effi",
        "sati_pos",
        "sati_neg",
        "sensa",
        "restrict",
        "enabling",
        "skill_tech_pro",
        "skill_progr",
        "lit_inf_pro",
        "lit_comm_pro",
        "lit_cont_pro",
        "kninf",
        "kncomm",
        "kncont",
        "skill_know_pro",
        "lit_overall_pro",
    ]

    TARGET_COLUMN = ["RISK101"]

    print("[INFO] Columns kept:", BASIC_INFO_COLUMNS + DERIVED_COLUMNS + TARGET_COLUMN)
    return BASIC_INFO_COLUMNS, DERIVED_COLUMNS, TARGET_COLUMN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reshape from wide to tall

    The original data has one row per student, with separate columns per wave. After this cell, the data will have one row per (student, wave) observation instead:
    """)
    return


@app.cell
def _(BASIC_INFO_COLUMNS, DERIVED_COLUMNS, TARGET_COLUMN, df_raw):
    _wave_columns = [c for c in BASIC_INFO_COLUMNS if c != "country"] + DERIVED_COLUMNS + TARGET_COLUMN

    _waves = []
    for _wave_no in (1, 2, 3):
        _frame = df_raw.select(["country", *[f"W{_wave_no}_{c}" for c in _wave_columns]])
        _frame.columns = ["country", *_wave_columns]
        _waves.append(_frame)

    df_long = pl.concat(_waves, how="vertical")

    print("[INFO] Tall shape:", df_long.shape)
    df_long
    return (df_long,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keep only rows with a valid target value

    Negative values throughout this dataset mean the data is not available for some reason (e.g. `-94` means "not asked"). Since a row without a `RISK101` value cannot be used as a labeled training example, we drop those rows now rather than later.
    """)
    return


@app.cell
def _(df_long):
    df_valid_target = df_long.filter(pl.col("RISK101") >= 0)

    print("[INFO] Rows before target filter:", df_long.height)
    print("[INFO] Rows after target filter:", df_valid_target.height)
    return (df_valid_target,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fix comma-decimals and standardize nulls

    Some numeric values in the raw CSV are encoded with a comma decimal separator (e.g. `,25` instead of `0.25`), which makes Polars infer those columns as strings. We cast every feature column to a proper `Float32` after replacing commas with periods.

    We also standardize missingness: instead of the old convention of a magic sentinel number, every remaining negative value (e.g. `-94` "not asked", `-98` "don't know") becomes a real Polars null. This matches the Titanic EDA notebook's approach of using nulls, and leaves the decision of how to impute them to the next (modeling) notebook.
    """)
    return


@app.cell
def _(BASIC_INFO_COLUMNS, DERIVED_COLUMNS, df_valid_target):
    _feature_columns = BASIC_INFO_COLUMNS + DERIVED_COLUMNS

    df_clean = (
        df_valid_target
        .with_columns([
            pl.col(c)
              .cast(pl.Utf8)
              .str.replace_all(",", ".")
              .cast(pl.Float32, strict=False)
            for c in _feature_columns
        ])
        .with_columns([
            pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in _feature_columns
        ])
        .with_columns(pl.col("RISK101").cast(pl.Int8))
    )

    df_clean
    return (df_clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Column-level work

    We now go through the remaining columns and place each one into one of two buckets: keep as categorical, or keep as numeric. `RISK101` is our target and is tracked separately.

    The rule is the same as in the Titanic EDA: unordered or low-cardinality grouping variables become categorical, while ordinal scales, proportions and counts stay numeric so their magnitude is preserved for the model.
    """)
    return


@app.cell
def _(BASIC_INFO_COLUMNS, DERIVED_COLUMNS, TARGET_COLUMN, df_clean):
    for _i, _col_name in enumerate(BASIC_INFO_COLUMNS + DERIVED_COLUMNS + TARGET_COLUMN, start=1):
        print(f"{_i}: {_col_name}")

    CATEGORICALS = set()
    NUMERICS = set()
    TARGET = list(TARGET_COLUMN)

    print("[INFO] Total rows:", df_clean.height)
    return CATEGORICALS, NUMERICS, TARGET


@app.cell(hide_code=True)
def _():
    def print_null_and_uniques(colname, dataframe, up_to=30):
        print(f"[INFO] The {colname} has null values: ", dataframe.select(colname).null_count().item())
        print(f"[INFO] Unique values has uniq values: ", dataframe.select(colname).n_unique())
        print(f"[INFO] Unique values are: ", dataframe.select(colname).unique().to_series().to_list()[:up_to])


    def plot_categorical_counts(
        dataframe,
        colname,
        color_col="RISK101",
        width=500,
    ):
        plot_df = (
            dataframe
            .group_by([colname, color_col])
            .agg(pl.len().alias("count"))
            .sort(colname)
        )

        return (
            plot_df.plot.bar(
                x=f"{colname}:N",
                y="count:Q",
                color=f"{color_col}:N",
            )
            .properties(width="container")
        )

    def plot_histograms(
        dataframe,
        columns,
        hue="RISK101",
        bins=20,
        discrete=False,
        columns_per_row=4,
    ):
        if isinstance(columns, str):
            columns = [columns]

        rows = math.ceil(len(columns) / columns_per_row)

        fig, axes = plt.subplots(
            rows,
            columns_per_row,
            figsize=(3.5 * columns_per_row, 3 * rows),
            squeeze=False,
        )

        for col, ax in zip(columns, axes.flat):
            sns.histplot(
                data=dataframe,
                x=col,
                hue=hue,
                bins=None if discrete else bins,
                discrete=discrete,
                multiple="layer",
                alpha=0.7,
                ax=ax,
            )
            ax.set_title(col)

        for ax in axes.flat[len(columns):]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig

    return plot_categorical_counts, plot_histograms, print_null_and_uniques


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. country

    `country` is stored as a small integer code, but semantically it is an unordered grouping variable (each value is a different country).

    **Verdict:** Keep as categorical, using the raw numeric code (no dictionary lookup table was available for readable country names).
    """)
    return


@app.cell
def _(CATEGORICALS, df_clean, plot_categorical_counts, print_null_and_uniques):
    _COL = "country"
    print_null_and_uniques(_COL, df_clean)

    CATEGORICALS.add(_COL)

    plot_categorical_counts(df_clean, _COL)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. GENDER

    A small categorical column with clear meaning and no extra parsing required.

    **Verdict:** Keep as-is as a categorical feature.
    """)
    return


@app.cell
def _(CATEGORICALS, df_clean, plot_categorical_counts, print_null_and_uniques):
    _COL = "GENDER"
    print_null_and_uniques(_COL, df_clean)

    CATEGORICALS.add(_COL)

    plot_categorical_counts(df_clean, _COL)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Age_year

    This is a discrete but genuinely ordinal value (age in years). Unlike `country`, the distance between values is meaningful.

    **Verdict:** Keep as numeric, mirroring the Titanic EDA's treatment of `age`.
    """)
    return


@app.cell
def _(NUMERICS, df_clean, plot_categorical_counts, print_null_and_uniques):
    _COL = "Age_year"
    print_null_and_uniques(_COL, df_clean)

    NUMERICS.add(_COL)

    plot_categorical_counts(df_clean, _COL)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. skill_progr

    This is a binary flag: whether the respondent reported any programming skill (from `SKILL1g`).

    **Verdict:** Keep as categorical, since it is a two-level flag rather than a magnitude.
    """)
    return


@app.cell
def _(CATEGORICALS, df_clean, plot_categorical_counts, print_null_and_uniques):
    _COL = "skill_progr"
    print_null_and_uniques(_COL, df_clean)

    CATEGORICALS.add(_COL)

    plot_categorical_counts(df_clean, _COL)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-7. friends, family, civic

    Mean Likert-scale scores (1-4) describing social relationships and civic engagement.

    **Verdict:** Keep all three as numeric.
    """)
    return


@app.cell
def _(NUMERICS, df_clean, plot_histograms, print_null_and_uniques):
    _cols = ["friends", "family", "civic"]

    for _col in _cols:
        print_null_and_uniques(_col, df_clean)
        NUMERICS.add(_col)

    plot_histograms(df_clean, _cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8-14. daily_activities, effi, sati_pos, sati_neg, sensa, restrict, enabling

    More derived wellbeing/behavior scales: a count of daily online activities (`daily_activities`), and several mean Likert-scale scores.

    **Verdict:** Keep all seven as numeric.
    """)
    return


@app.cell
def _(NUMERICS, df_clean, plot_histograms, print_null_and_uniques):
    _cols = [
        "daily_activities", "effi", "sati_pos", "sati_neg",
        "sensa", "restrict", "enabling",
    ]

    for _col in _cols:
        print_null_and_uniques(_col, df_clean)
        print()
        NUMERICS.add(_col)

    plot_histograms(df_clean, _cols, bins=6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15-19. skill_tech_pro, lit_inf_pro, lit_comm_pro, lit_cont_pro, lit_overall_pro, skill_know_pro

    Proportions (0-1) of skill/knowledge items answered at a high level or correctly.

    **Verdict:** Keep all as numeric.
    """)
    return


@app.cell
def _(NUMERICS, df_clean, plot_histograms, print_null_and_uniques):
    _cols = ["skill_tech_pro", "lit_inf_pro", "lit_comm_pro", "lit_cont_pro", "lit_overall_pro", "skill_know_pro"]

    for _col in _cols:
        print_null_and_uniques(_col, df_clean)
        print()
        NUMERICS.add(_col)

    plot_histograms(df_clean, _cols, bins=5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 20-22. kninf, kncomm, kncont

    Counts of correct answers (0-2) on short knowledge quizzes about information, communication and content skills.

    **Verdict:** Keep as numeric — these are small ordinal counts, similar in spirit to Titanic's `family_size`.
    """)
    return


@app.cell
def _(NUMERICS, df_clean, plot_histograms, print_null_and_uniques):
    _cols = ["kninf", "kncomm", "kncont"]

    for _col in _cols:
        print_null_and_uniques(_col, df_clean)
        print()
        NUMERICS.add(_col)

    plot_histograms(df_clean, _cols, discrete=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 23. RISK101

    This is our prediction target: whether the respondent experienced cyberhate in the past year. It has no missing values left, since we already filtered those rows out.

    **Verdict:** Keep as the `target`. Any class-imbalance handling belongs downstream.
    """)
    return


@app.cell
def _(df_clean, plot_categorical_counts, print_null_and_uniques):
    _COL = "RISK101"
    print_null_and_uniques(_COL, df_clean)

    plot_categorical_counts(df_clean, _COL, color_col="GENDER")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation overview

    With ~20 numeric columns, a full scatter matrix (as used for Titanic) would be too dense to read. Instead we use a Seaborn correlation heatmap over the numeric features and the target.
    """)
    return


@app.cell
def _(NUMERICS, TARGET, df_clean):
    _corr = df_clean.select(sorted(NUMERICS) + TARGET).to_pandas().corr()

    _fig, _ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Write to disk

    We are now ready to write the semantic dataset to disk. The sidecar JSON file records which columns are categorical, numeric, and which column is the target, so the next notebook can build a model pipeline without repeating the EDA work.
    """)
    return


@app.cell
def _(CATEGORICALS, NUMERICS, TARGET, df_clean):
    metadata = {
        "categoricals": sorted(CATEGORICALS),
        "numerics": sorted(NUMERICS),
        "target": list(TARGET),
    }
    print(metadata)

    df_write = df_clean.select(sorted(CATEGORICALS) + sorted(NUMERICS) + list(TARGET))
    df_write
    return df_write, metadata


@app.cell
def _(YSKILLS_META_FILE, YSKILLS_OUT_FILE, df_write, metadata):
    df_write.write_parquet(YSKILLS_OUT_FILE)

    with open(YSKILLS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Wrote data to: {YSKILLS_OUT_FILE}")
    print(f"[INFO] Wrote metadata to: {YSKILLS_META_FILE}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion

    This notebook produced a semantic ySKILLS dataset intended to be consumed by the modeling notebook.

    What we did:

    * merged the three survey waves into a single tall table
    * dropped rows without a valid `RISK101` label
    * cherry-picked the dictionary-recommended derived columns, dropping known-redundant duplicates
    * fixed comma-decimal string encoding
    * standardized all missingness (blanks and negative "not asked"/"don't know" codes) to real nulls

    In the next notebook, `714_y_skills_logreg.py`, we will load the Parquet file and train a model.
    """)
    return


if __name__ == "__main__":
    app.run()
