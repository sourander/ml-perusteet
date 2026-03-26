import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl

    from math import prod, log
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.compose import ColumnTransformer


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Naive Bayes from Scratch

    Now you know (from previous Marimo Notebook) how Count Vectorizing and TF-IDF works. This allows us to simply utilize a ready-made solution. Let's use CountVectorizer from scikit-learn. Using this data, we will create our own version of Multinomial Naive Bayes.

    ## Dataset

    Our dataset has column raw_text. A separate dataset `y` contains the labels.
    """)
    return


@app.cell
def _():
    corpus_pos = [
        "bad bad now",
        "bad now yes",
        "yes no no",
    ]

    corpus_neg = [
        "bad good no",
        "no no no",
        "bad yes no",
    ]
    corpus = corpus_pos + corpus_neg
    labels = [1] * len(corpus_pos) + [0] * len(corpus_neg)

    X_raw = pl.DataFrame(
        {"raw_text": corpus},
        schema={
            "raw_text": pl.String,
        },
    )

    y = pl.DataFrame({"spam": labels}, schema={"spam": pl.Int8})
    X_raw.hstack(y)
    return X_raw, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Count words into new columns

    We will also keep the `raw_text` field. This will be used for printing.
    """)
    return


@app.cell
def _(X_raw):
    ct = ColumnTransformer(
        [("text", CountVectorizer(), "raw_text")],
        sparse_threshold=False,
        verbose_feature_names_out=False,
    )

    X_sparse = ct.fit_transform(X_raw)
    sparse_schema = {x: pl.Int8 for x in ct.get_feature_names_out()}

    X_ct = pl.DataFrame(X_sparse, schema=sparse_schema)

    X = X_raw.hstack(X_ct)
    X
    return X, ct, sparse_schema


@app.class_definition
class NaiveBayesPolars:
    def __init__(self):
        # Friendly names
        self.LABELS = (0, 1)
        self.LABEL_NAMES = {0: "not spam", 1: "spam"}

        # Fitted statistics
        self.n = 0              # N
        self.n_class = {}       # N(y)
        self.priors = {}        # P(y)
        self.feature_cols = []  # ["egg", "ham", ..., "spam"]
        self.vocab_size = 0     # len(feature_cols)

        # Per-class counts and probabilities
        self.word_counts = {}
        self.total_words_in_class = {}
        self.word_likelihoods = {}

        # A local copy of original DataFrame for filtering
        # is created during fit().
        # self.df

    def fit(self, X: pl.DataFrame, y: pl.DataFrame, alpha: float = 0.5):
        """
        X: Polars DataFrame with:
           - one column 'raw_text'
           - remaining columns are word-count features
        y: Polars DataFrame with one column 'spam'
        """

        # Vocabulary level
        self._set_feature_columns(X)

        # Full dataset level
        self.df = X.with_columns(y["spam"])
        self.n = self.df.height

        # Per label level
        self._count_classes()
        self._calculate_priors()
        self._count_words_per_class()
        self._calculate_word_likelihoods(alpha)

        return self

    def _set_feature_columns(self, X: pl.DataFrame):
        """Store vocabulary columns (everything except raw_text)."""
        self.feature_cols = [c for c in X.columns if c != "raw_text"]
        self.vocab_size = len(self.feature_cols)

    def _count_classes(self):
        """Count how many documents belong to each label."""
        for label in self.LABELS:
            self.n_class[label] = self._filter_by_label(label).height

    def _calculate_priors(self):
        """Calculate class prior probabilities P(y)."""
        for label in self.LABELS:
            self.priors[label] = self.n_class[label] / self.n

    def _count_words_per_class(self):
        """
        Count total word occurrences per feature for each class.

        Example:
            word_counts[1]["bad"] = total count of word 'bad'
            in all spam documents.
        """
        for label in self.LABELS:
            class_df = self._filter_by_label(label)

            # Sum each word column over all documents in this class
            counts_row = class_df.select(self.feature_cols).sum().row(0, named=True)

            self.word_counts[label] = counts_row
            self.total_words_in_class[label] = sum(counts_row.values())

    def _calculate_word_likelihoods(self, alpha: float):
        """
        Calculate smoothed likelihoods:

            P(word | class) =
                (count(word, class) + alpha)
                /
                (total_words_in_class[class] + alpha * vocab_size)
        """
        for label in self.LABELS:
            self.word_likelihoods[label] = {}

            denominator = self.total_words_in_class[label] + alpha * self.vocab_size

            for word in self.feature_cols:
                numerator = self.word_counts[label][word] + alpha
                self.word_likelihoods[label][word] = numerator / denominator

    def _filter_by_label(self, label: int) -> pl.DataFrame:
        """Return only rows whose class label is the given label."""
        return self.df.filter(pl.col("spam") == label)


    def predict_one_log(self, row: dict, verbose: bool = True):
        scores = {}

        for label in self.LABELS:
            score = log(self.priors[label])

            for word in self.feature_cols:
                count = int(row[word])
                if count > 0:
                    score += count * log(self.word_likelihoods[label][word])

            scores[label] = score

        prediction = max(scores, key=scores.get)

        if verbose:
            print(f"Text: {row['raw_text']}")
            for label in self.LABELS:
                print(f"  log-score for {self.LABEL_NAMES[label]:>8}: {scores[label]:.6f}")
            print(f"-> Prediction: {self.LABEL_NAMES[prediction]}")
            print()

        return prediction


    def predict(self, X: pl.DataFrame, verbose: bool = True):
        """
        Predict all rows in X.
        Returns a Python list of labels.
        """

        predictions = []

        for row in X.iter_rows(named=True):
            pred = self.predict_one_log(row, verbose=verbose)
            predictions.append(pred)

        return predictions

    def show_training_statistics(self):
        print(f"N documents: {self.n}")
        print(f"Vocabulary: {self.feature_cols}")
        print()

        for label in self.LABELS:
            print(f"Class {label} ({self.LABEL_NAMES[label]}):")
            print(f"  N(class): {self.n_class[label]}")
            print(f"  Prior:    {self.priors[label]:.4f}")
            print(f"  Total words in class: {self.total_words_in_class[label]}")
            print(f"  Word counts: {self.word_counts[label]}")
            print("  Likelihoods:")
            for word in self.feature_cols:
                print(f"    P({word}|{label}) = {self.word_likelihoods[label][word]:.4f}")
            print()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train
    """)
    return


@app.cell
def _(X, y):
    nb = NaiveBayesPolars()
    nb.fit(X, y, alpha=0.5)
    nb.show_training_statistics()
    return (nb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Accuracy on Train Set
    """)
    return


@app.cell
def _(X, nb, y):
    preds = nb.predict(X, verbose=False)
    targets = y.to_dict()["spam"]

    correct = sum([1 for y_hat, y in zip(preds, targets) if y_hat == y])
    print(f"Accuracy: {correct / y.height:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test

    ### Create Test Dataset
    """)
    return


@app.cell
def _(ct, sparse_schema):
    test_corpus_pos = [
        "bad bad NEW", # a new word
        "bad now yes",
        "DONT YET YELLOW SNOW",
    ]

    test_corpus_neg = [
        "bad good ANOTHER", # another unseen word
        "no no no",
        "bad yes no",
        "ALL NEW WORDS",    # sentence containing only unseen words
    ]
    test_corpus = test_corpus_pos + test_corpus_neg
    test_labels = [1] * len(test_corpus_pos) + [0] * len(test_corpus_neg)

    X_test_raw = pl.DataFrame(
        {"raw_text": test_corpus},
        schema={
            "raw_text": pl.String,
        },
    )

    y_test = pl.DataFrame({"spam": test_labels}, schema={"spam": pl.Int8})
    X_test_sparse = ct.transform(X_test_raw)
    X_test_ct = pl.DataFrame(X_test_sparse, schema=sparse_schema)
    X_test = X_test_raw.hstack(X_test_ct)
    X_test
    return X_test, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Accuracy on Test Set
    """)
    return


@app.cell
def _(X_test, nb, y_test):
    preds_test = nb.predict(X_test, verbose=False)

    targets_test = y_test.to_dict()["spam"]

    correct_test = sum([1 for y_hat, y in zip(preds_test, targets_test) if y_hat == y])
    print(f"Accuracy: {correct_test / y_test.height:.2f}")
    return


if __name__ == "__main__":
    app.run()
