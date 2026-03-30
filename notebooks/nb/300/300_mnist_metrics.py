import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt
    import numpy as np

    from sklearn import datasets, metrics, svm, linear_model
    from sklearn.model_selection import train_test_split


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SVC and Metrics

    This Notebook includes the calculations used for presenting the numbers in the lesson **Luokittelumallin suorituskyky**. You can play around with the cells, and also, there is a task in the very last cell that you can use to challenge yourself.
    """)
    return


@app.cell
def _():
    digits = datasets.load_digits()

    # Convert into binary problem. 
    y = digits.target == 3

    # Fit and predict
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, 
        y, 
        test_size=0.3, 
        random_state=160 # Seed 
    )

    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(metrics.classification_report(
        y_test, 
        y_pred, 
        target_names=['Luku N', 'Luku 3'] # 
    ))
    return X_test, clf, y_pred, y_test


@app.cell
def _(y_pred, y_test):
    #for target, predicted in zip(y_test, y_pred)[:10]:
    #    print(target, predicted)

    pl.DataFrame(data=[y_test, y_pred], schema={"y_test": pl.Boolean, "y_pred": pl.Boolean})
    return


@app.cell
def _(y_pred, y_test):
    def tn_fp_fn_tp(y_test, y_pred):
        # Init
        TN, FP, FN, TP = 0, 0, 0, 0

        # Count
        for pair in zip(y_test, y_pred):
            match pair:
                case (1, 1):
                    TP += 1
                case (0, 0):
                    TN += 1
                case (0, 1):
                    FP += 1
                case (1, 0):
                    FN += 1

        return ((TN, FP), (FN, TP))

    (TN, FP), (FN, TP) = tn_fp_fn_tp(y_test, y_pred)
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    return FN, FP, TN, TP, tn_fp_fn_tp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrix
    """)
    return


@app.cell
def _(tn_fp_fn_tp, y_pred, y_test):
    ours = tn_fp_fn_tp(y_test, y_pred)
    theirs = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    ours == theirs
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy
    """)
    return


@app.cell
def _(FN, FP, TN, TP, y_pred, y_test):
    acc = (TP + TN) / (TP + TN + FP + FN)
    assert metrics.accuracy_score(y_test, y_pred) == acc

    acc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recall
    """)
    return


@app.cell
def _(FN, TP, y_pred, y_test):
    recall = TP / (TP + FN)
    assert metrics.recall_score(y_test, y_pred) == recall

    recall
    return (recall,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specificity
    """)
    return


@app.cell
def _(FP, TN, y_pred, y_test):
    specificity = TN / (TN + FP)
    assert metrics.recall_score(y_test, y_pred, pos_label=0) == specificity

    specificity
    return (specificity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Balanced Accuracy
    """)
    return


@app.cell
def _(recall, specificity, y_pred, y_test):
    balanced_acc = (recall+specificity) / 2
    assert metrics.balanced_accuracy_score(y_test, y_pred) == balanced_acc

    balanced_acc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ROC work begins
    """)
    return


@app.cell
def _(X_test, clf, y_test):
    # Use probability of the positive class (usually class 1)
    y_score = clf.decision_function(X_test)

    # Compute ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

    # Put into a DataFrame for Altair
    roc_df = pl.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr,
        "Threshold": thresholds
    }).with_columns(
        Threshold=pl.when(pl.col("Threshold").is_infinite())
        .then(None)
        .otherwise(pl.col("Threshold"))
    )


    # ROC line
    roc_line = alt.Chart(roc_df).mark_line().encode(
        x=alt.X("False Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("True Positive Rate:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("False Positive Rate:Q", format=".3f"),
            alt.Tooltip("True Positive Rate:Q", format=".3f"),
            alt.Tooltip("Threshold:Q", format=".3f")
        ]
    )

    # Diagonal reference line (random classifier)
    diag_df = pl.DataFrame({
        "x": [0, 1],
        "y": [0, 1]
    })

    diag_line = alt.Chart(diag_df).mark_line(strokeDash=[5, 5], color="gray").encode(
        x="x:Q",
        y="y:Q"
    )

    # Combine chart
    _chart = (roc_line + diag_line).properties(
        width=500,
        height=400,
    )

    _chart
    return fpr, roc_df, tpr, y_score


@app.cell
def _(roc_df):
    roc_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Histogram of decision scores
    """)
    return


@app.cell(hide_code=True)
def _(y_score, y_test):

    _step = 0.1

    _df = pl.DataFrame({
        "y_score": y_score,
        "y_true": y_test,
    })

    _step = 0.1

    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X(
                field="y_score",
                type="quantitative",
                bin={"step": _step},
                title="Decision score"
            ),
            y=alt.Y(
                aggregate="count",
                type="quantitative",
                title="Count"
            ),
            color=alt.Color(
                field="y_true",
                type="nominal",
                title="True label"
            ),
            tooltip=[
                alt.Tooltip(field="y_score", type="quantitative", bin={"step": _step}, title="Score bin"),
                alt.Tooltip(field="y_true", type="nominal", title="True label"),
                alt.Tooltip(aggregate="count", type="quantitative", title="Count"),
            ]
        )
        .properties(
            height=290,
            width="container"
        )
        .configure_axis(
            grid=False
        )
    )

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AUC
    """)
    return


@app.cell
def _(fpr, tpr):
    # Using NumPy
    np.sum(tpr[1:] * np.diff(fpr))
    return


@app.cell
def _(fpr, tpr):
    # Using Scikit
    metrics.auc(fpr, tpr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PR AUC work begins
    """)
    return


@app.cell
def _(y_score, y_test):
    precisions, recalls, pr_thresholds = metrics.precision_recall_curve(y_test, y_score)
    return pr_thresholds, precisions, recalls


@app.cell
def _(pr_thresholds, precisions, recalls):
    prc_df = pl.DataFrame({
        "Recall": recalls,
        "Precision": precisions,
        "Threshold": list(pr_thresholds) + [None]
    })

    _chart = alt.Chart(prc_df).mark_line().encode(
        x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("Recall:Q", format=".3f"),
            alt.Tooltip("Precision:Q", format=".3f"),
            alt.Tooltip("Threshold:Q", format=".3f")
        ]
    ).properties(
        width=500,
        height=400,
    )

    _chart

    return


@app.cell
def _(precisions, recalls):
    # Calculate the AUC too
    metrics.auc(recalls, precisions)
    return


@app.cell
def _(y_score, y_test):
    # Tai sitten
    metrics.average_precision_score(y_test, y_score)
    return


@app.cell
def _(precisions, recalls):
    -np.trapezoid(precisions, recalls)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extra Challenge

    ROC and PR curves are boring when the model is near-perfect. Try switching to a more challenging dataset. One option is to generate your own, like so:

    ```
    X, y = datasets.make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=4,
        n_redundant=6,
        n_repeated=0,
        n_classes=2,
        class_sep=1.0,   # smaller = harder; try adjusting up and down
        random_state=160
    )

    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=160
    )
    ```

    Obviously, this is dummy data, so classes 0 and 1 mean nothing. Let `1` be `is_banana`, and `0` a `not_banana`.
    """)
    return


if __name__ == "__main__":
    app.run()
