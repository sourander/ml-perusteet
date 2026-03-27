import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score, classification_report


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classifying Hate Speech with Naive Bayes

    Remember, your task is to classify a binary problem: a tweet either is or is not hate speech.

    ## Getting Data

    You need to download the data from the repository: [gh:t-davison/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language)
    """)
    return


@app.cell
def _():
    # Add relative path to the file
    WHERE_YOU_DOWNLOADED_THE_FILE = "data/hatespeech/labeled_data.csv"

    SCORING="accuracy"
    return SCORING, WHERE_YOU_DOWNLOADED_THE_FILE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading DataFrame

    The first lines of the dataset look like this:

    ```csv
    ,count,hate_speech,offensive_language,neither,class,tweet
    0,3,0,0,3,2,!!! RT @mayasolovely: some text here
    1,3,0,3,0,1,!!!!! RT @mleew17: more text here
    2,3,0,3,0,1,!!!!!!! RT @UrKindOfBrand Dawg!!!! even more text
    ```

    Load the fields you need into a DataFrame. Then, do some required EDA (Exploratory Data Analysis), after which you should:

    - process data as required by the chosen model (Naive Bayes)
    - split to train and test
    - fit the model
    - evaluate the model, and
    - report results.
    """)
    return


@app.cell
def _(WHERE_YOU_DOWNLOADED_THE_FILE):
    schema = {
        "class": pl.Int8,
        "tweet": pl.Utf8
    }

    class_map = {
        0: 1,    # orig 0 is hate speech
        1: None, # orig 1 is offensive
        2: 0     # orig 2 is neither
    }

    df = pl.read_csv(WHERE_YOU_DOWNLOADED_THE_FILE, columns=list(schema.keys()), schema_overrides=schema)
    df = df.rename({"class": "label"}) # class has specific meaning in Python. Let's rename.

    df = df.with_columns(
        label=pl.col("label").replace(class_map)
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Be wary of the class imbalance
    """)
    return


@app.cell
def _(df):
    # replace _df with your data source
    _chart = (
        alt.Chart(df.group_by("label").len())
        .mark_bar()
        .encode(
            x=alt.X(field='label', type='nominal'),
            y=alt.Y(field='len', type='quantitative'),
            tooltip=[
                alt.Tooltip(field='label'),
                alt.Tooltip(field='len')
            ]
        )
        .properties(
            height=290,
            width='container'
        )
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clean Up Junk

    You can clean up the data as you see fit. This quick clean-up was generated using GPT-5.4 to help with Regex syntax.
    """)
    return


@app.cell
def _(df):
    # Drop nulls. These will be the labels that were chosen to be dropped.
    df_filtered = df.drop_nulls()

    regex_html = r"(?:&amp;)+(?:#\d+|[a-zA-Z]+);|&(?:#\d+|[a-zA-Z]+);"

    df_cleaned = df_filtered.with_columns(
        pl.col("tweet")
        .str.replace_all(r"https?://\S+", "")  # URLs
        .str.replace_all(r"@\w+", "")  # @mentions
        .str.replace_all(regex_html, "")  # plain + nested HTML entities
        .str.replace_all(r"[!\"':\-.]", "")  # remove !, ", ', :, -, .
        .str.replace_all(r"\s+", " ")  # collapse whitespace
        .str.strip_chars()  # trim ends
    )

    df_cleaned
    return (df_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## X and y
    """)
    return


@app.cell
def _(df_cleaned):
    X = df_cleaned.get_column("tweet")
    y = df_cleaned.get_column("label")
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Test Split

    The test set is held out for **final evaluation only**. During model iteration, we use 5-fold cross-validation on the training data to compare models without touching the test set.
    """)
    return


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 1: Default Settings

    Let's start with the simplest possible setup: `CountVectorizer` with default settings and `MultinomialNB` with default settings. This is our **baseline** model.
    """)
    return


@app.cell
def _(SCORING, X_train, y_train):
    pipeline1 = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB()),
    ])

    cv_scores1 = cross_val_score(pipeline1, X_train, y_train, cv=5, scoring=SCORING)
    print(f"CV scores: {cv_scores1}")
    print(f"Mean {SCORING}: {cv_scores1.mean():.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cross-validation scores above show the **F1 score** for each of the 5 folds. F1 is the harmonic mean of precision and recall — especially important when classes are imbalanced. Try to find out what precision and recall each mean and why they matter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 2: Adding Bigrams to Vectorizer

    In Model 1, the vectorizer only considered individual words (unigrams). Now let's try `ngram_range=(1, 2)`.

    **Note:** If you do end up using this teacher's choice of adjusting the N-gram settings, you should find out what N-gram means. Explaining this in your learning diary would be a good idea.
    """)
    return


@app.cell
def _(SCORING, X_train, y_train):
    pipeline2 = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB()),
    ])

    cv_scores2 = cross_val_score(pipeline2, X_train, y_train, cv=5, scoring=SCORING)
    print(f"CV scores: {cv_scores2}")
    print(f"Mean {SCORING}: {cv_scores2.mean():.4f}")
    return (pipeline2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What Next?

    You have now trained two models and compared their cross-validation scores. This **iterative process** is how you improve a model step by step. Each time, change **one thing** and observe the effect. Here are some ideas for your next experiments:

    - Try `CountVectorizer(binary=True)` — count only whether a word appears, not how many times.
    - Try different `min_df` or `max_df` values — these control which words are included in the vocabulary.
    - Try changing the smoothing parameter: `MultinomialNB(alpha=0.5)` instead of the default `1.0`.
    - Read the CountVectorizer, TfidfVectorizer, MultinomialNB and ComplementNB docs. Figure out what hyperparameters would be reasonable to try out.
    - Play around with the `class_map`. What to do with the offensive class?

    Keep iterating until you are satisfied with your cross-validation results. Once you have found the best configuration, proceed to the final evaluation below.

    /// note | Remember
    This is your Notebook now. Change whatever you want. This is like an assignment in primary school where the teacher is expecting a single correct answer.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final Evaluation

    Once you are satisfied with the cross-validation results, choose the best configuration and evaluate it **once** on the held-out test set. Replace the pipeline definition below with your best settings.


    /// warning | Keep in mind!
    This is NOT a competition. Getting high scores by copy-pasting code from an AI Agent that you do not understand is **not the purpose** of this exercise. Getting F1 score of 0.95 will not guarantee you a good grade. Increasing your own understanding is the primary goal. If you do a thing $Z$ and it helps you learn, then $Z$ is likely a good thing to do.

    Having that said, teacher trained a model using GridCV to find good settings for a chosen vectorizer and an estimator. Using those settings, the test evaluation (out-of-sample) results results were the following...
    ///

    ```
    Test set report:
                  precision    recall  f1-score   support

               0       0.94      0.96      0.95       833
               1       0.88      0.84      0.86       286

        accuracy                           0.93      1119
       macro avg       0.91      0.90      0.90      1119
    weighted avg       0.93      0.93      0.93      1119
    ```
    """)
    return


@app.cell
def _(X_test, X_train, pipeline2, y_test, y_train):
    # Toggle this ONLY AS THE LAST STEP. You should not tune your
    # model against the test set. This is supposed to be fully unseen
    I_AM_SURE_I_HAVE_CHOSEN_MY_HYPERPARAMETERS = True

    if I_AM_SURE_I_HAVE_CHOSEN_MY_HYPERPARAMETERS:
        best_pipeline = pipeline2 # <- CHANGE THIS
    
        best_pipeline.fit(X_train, y_train)
        _y_pred = best_pipeline.predict(X_test)
    
        print(f"Accuracy: {accuracy_score(y_test, _y_pred):.4f}")
        print()
        print(classification_report(y_test, _y_pred))
    return


if __name__ == "__main__":
    app.run()
