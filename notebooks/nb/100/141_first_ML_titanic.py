import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    import polars as pl
    import altair as alt
    import numpy as np
    import math

    from pathlib import Path

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, train_test_split

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        balanced_accuracy_score
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Titanic Survival Prediction

    Let's start by loading the Parquet file and the sidecar metadata file.

    ## Data Loading
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
    TARGET = meta["target"]

    print("[INFO] Data columns:", df.columns)
    print("[INFO] CATEGORICALS:", CATEGORICALS)
    print("[INFO] NUMERICS:", NUMERICS)
    print("[INFO] TARGET:", TARGET)
    return CATEGORICALS, NUMERICS, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Test Split
    """)
    return


@app.cell
def _(df):
    X = df.drop("survived")
    y = df.get_column("survived")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("[INFO] Lines in training set", X_train.height)
    print("[INFO] Lines in testing set", X_test.height)

    X_train
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One Hot Encoding

    This encoder CAN be placed into Pipeline further down, but let's leave it here. If we would want, we could wrap this e.g. in a loop and train with various `min_frequency` values.
    """)
    return


@app.cell
def _(CATEGORICALS, NUMERICS, X_test, X_train):
    def one_hot_encode(X):
        encoded = ohe.transform(X.select(CATEGORICALS))
        return pl.concat([
            X.select(NUMERICS),
            pl.DataFrame(encoded, schema=ohe_feature_names)
        ], how="horizontal")

    # Pre-fit OneHotEncoder on training data only, outside of GridSearch
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.1, drop="if_binary")
    ohe.fit(X_train.select(CATEGORICALS))

    ohe_feature_names = list(ohe.get_feature_names_out(CATEGORICALS))

    X_train_encoded = one_hot_encode(X_train)
    X_test_encoded = one_hot_encode(X_test)

    print("[INFO] Encoded training set shape:", X_train_encoded.shape)
    print("[INFO] OHE feature names:")
    for feat_name in ohe_feature_names:
        print("  - ", feat_name)
    return X_test_encoded, X_train_encoded, one_hot_encode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training with Grid Search

    Instead of manually trying out all sorts of hyperparameter combinations, we will utilize Grid Search. To do that, we need to set up pipeline. Then, we will use parameter grid to swap out pieces out of that pipeline: this will total into hundreds of ML model trainings, but since our dataset it tiny, training will not take more than some seconds (assuming you have multiple CPUs available for parallelism).

    ## Define Pipelines
    """)
    return


@app.cell
def _(NUMERICS):
    pipe_nums = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('nums', pipe_nums, NUMERICS),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    param_grid = [
        {
            'classifier': [
                LogisticRegression(),
            ],
            'classifier__C': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            'preprocessor__nums__imputer': [
                SimpleImputer(strategy='mean'),
                SimpleImputer(strategy='median'),
                KNNImputer(n_neighbors=3, weights='uniform'),
                KNNImputer(n_neighbors=5, weights='uniform'),
                KNNImputer(n_neighbors=3, weights='distance'),
                KNNImputer(n_neighbors=5, weights='distance'),
            ],
            'preprocessor__nums__scaler': [StandardScaler()],
        },
        {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 5, 10],
            'preprocessor__nums__imputer': [
                SimpleImputer(strategy='mean'),
                SimpleImputer(strategy='median'),
                KNNImputer(n_neighbors=3, weights='uniform'),
                KNNImputer(n_neighbors=5, weights='uniform'),
                KNNImputer(n_neighbors=3, weights='distance'),
                KNNImputer(n_neighbors=5, weights='distance'),
            ],
            'preprocessor__nums__scaler': ['passthrough'],
        },
        {
            'classifier': [SVC(probability=True, random_state=42)],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['rbf', 'linear'],
            'preprocessor__nums__imputer': [
                SimpleImputer(strategy='mean'),
                SimpleImputer(strategy='median'),
                KNNImputer(n_neighbors=5, weights='uniform'),
            ],
            'preprocessor__nums__scaler': [StandardScaler()],
        },
    ]
    return param_grid, pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the Search
    """)
    return


@app.cell
def _(X_train_encoded, param_grid, pipeline, y_train):
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit
    grid_search.fit(X_train_encoded, y_train)

    # Best results
    print("Best score:", grid_search.best_score_)
    print("Best params:", grid_search.best_params_)
    return (grid_search,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fetch Best per Model Type

    ## Handle the Results Dict
    """)
    return


@app.cell
def _(grid_search):
    df_results = pl.DataFrame(grid_search.cv_results_, strict=False)

    # Get the classifier name as a column
    df_results = df_results.with_columns(
        pl.col("params").map_elements(
            lambda p: type(p["classifier"]).__name__,
            return_dtype=pl.Utf8
        ).alias("classifier_name")
    )

    # View the DataFrame if you find it helpful
    # df_results.select(["classifier_name", "mean_test_score"])
    return (df_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Density Chart
    """)
    return


@app.cell(hide_code=True)
def _(df_results):
    density_chart = (
        alt.Chart(df_results)
        .transform_density(
            "mean_test_score",
            groupby=["classifier_name"],
            as_=["mean_test_score", "density"]
        )
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X("mean_test_score:Q", title="Mean CV test score"),
            y=alt.Y("density:Q", stack=None, title="Density"),
            color=alt.Color("classifier_name:N", title="Classifier"),
            tooltip=[
                alt.Tooltip("classifier_name:N", title="Classifier"),
                alt.Tooltip("mean_test_score:Q", format=".4f", title="Score"),
                alt.Tooltip("density:Q", format=".4f", title="Density")
            ]
        )
        .properties(
            height=290,
            width="container"
        )
        .configure_axis(grid=False)
    )

    density_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract One per Family
    """)
    return


@app.cell
def _(df_results):
    df_best_per_family = (
        df_results
        .sort(["classifier_name", "mean_test_score"], descending=[False, True])
        .unique(subset=["classifier_name"], keep="first", maintain_order=True)
        .select(["classifier_name", "mean_test_score", "std_test_score", "params"])
    )

    df_best_per_family
    return (df_best_per_family,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit the One per Family

    The Grid Search remember only the best model as a trained model. All other are kept only as parameter values. Thus, we will need to fit at least the others except the winner. Easy option: quickly retrain all in a loop.
    """)
    return


@app.cell
def _(X_train_encoded, df_best_per_family, grid_search, y_train):
    best_models = {}

    for row in df_best_per_family.iter_rows(named=True):
        clf_name = row["classifier_name"]
        best_params = row["params"]

        model = clone(grid_search.estimator)
        model.set_params(**best_params)
        model.fit(X_train_encoded, y_train)

        best_models[clf_name] = model
    return (best_models,)


@app.cell
def _():
    # Uncomment either to see what Pipeline in a visual format (thanks to Marimo)
    # best_models["LogisticRegression"]
    # best_models["RandomForestClassifier"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test
    """)
    return


@app.cell
def _(X_test_encoded, best_models, pipeline, y_test):
    rows = []

    for model_name, _pipeline in best_models.items():
        y_pred = _pipeline.predict(X_test_encoded)

        metric_values = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        }

        if hasattr(pipeline, "predict_proba"):
            y_proba = _pipeline.predict_proba(X_test_encoded)[:, 1]
            metric_values["ROC AUC"] = roc_auc_score(y_test, y_proba)
            metric_values["PR AUC (Average Precision)"] = average_precision_score(y_test, y_proba)

        for metric_name, metric_value in metric_values.items():
            rows.append({
                "model": model_name,
                "metric": metric_name,
                "value": metric_value,
            })

    metrics_df = pl.DataFrame(rows)
    # View if it helps:
    # metrics_df
    return (metrics_df,)


@app.cell(hide_code=True)
def _(metrics_df):
    _chart = (
        alt.Chart(metrics_df)
        .mark_bar()
        .encode(
            x=alt.X('metric:N', title='Metric'),
            xOffset=alt.XOffset('model:N'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('model:N', title='Model'),
            tooltip=[
                alt.Tooltip('metric:N', title='Metric'),
                alt.Tooltip('model:N', title='Model'),
                alt.Tooltip('value:Q', title='Value', format='.3f'),
            ]
        )
        .properties(
            height=290,
            width='container'
        )
        .configure_axis(grid=False)
    )

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Importances
    """)
    return


@app.cell
def _(best_models):
    _rows = []

    for _model_name, _fitted_pipeline in best_models.items():
        _feature_names = [
            name.removeprefix("remainder__")
            for name in _fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
        ]
        _classifier = _fitted_pipeline.named_steps["classifier"]

        # Tree algorithms call them feature importances
        if hasattr(_classifier, "feature_importances_"):
            _importance_values = _classifier.feature_importances_
            _metric_name = "Feature_Importance"
        # Linear models, like Logaritmic Regression, call them coefficients
        elif hasattr(_classifier, "coef_"):
            _importance_values = _classifier.coef_[0]
            _metric_name = "Coefficient"
        # Panic route for any other situation
        else:
            _importance_values = np.zeros(len(_feature_names))
            _metric_name = "Value_Not_Available"

        for _feat, _val in zip(_feature_names, _importance_values):
            _rows.append({
                "model": _model_name,
                "feature": _feat,
                "metric": _metric_name,
                "value": float(_val),
                "absolute_impact": float(np.abs(_val)),
            })

    importance_df = pl.DataFrame(_rows).sort("absolute_impact", descending=True)
    # View if it helps:
    # importance_df
    return (importance_df,)


@app.cell(hide_code=True)
def _(importance_df):
    _chart = (
        alt.Chart(importance_df)
        .mark_bar()
        .encode(
            x=alt.X(
                'feature:N',
                sort=alt.EncodingSortField(
                    field='absolute_impact',
                    op='sum',              # or 'mean', 'max'
                    order='descending'
                ),
                axis=alt.Axis(labelAngle=-45)
            ),
            xOffset=alt.XOffset('model:N'),
            y=alt.Y('value:Q'),
            color=alt.Color('model:N'),
            tooltip=[
                alt.Tooltip('feature:N'),
                alt.Tooltip('value:Q', format=',.2f'),
                alt.Tooltip('absolute_impact:Q', format=',.2f'),
                alt.Tooltip('model:N')
            ]
        )
        .properties(
            height=290,
            width='container'
        )
        .configure_axis(grid=False)
    )

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test Subject X
    """)
    return


@app.cell(hide_code=True)
def _():
    def unique_options(df: pl.DataFrame, col: str) -> list[str]:
        values = (
            df.select(
                pl.col(col)
                .drop_nulls()
                .cast(pl.Utf8)
                .unique()
                .sort()
            )
            .to_series()
            .to_list()
        )
        return values

    def mode_option(df: pl.DataFrame, col: str):
        # Most common non-null value
        out = (
            df.select(
                pl.col(col)
                .drop_nulls()
                .mode()
                .first()
                .cast(pl.Utf8)
                .alias("mode_value")
            )
            .to_series()
            .to_list()
        )
        return out[0] if out else None

    def int_slider_stats(df: pl.DataFrame, col: str):
        row = (
            df.select(
                pl.col(col).drop_nulls().min().cast(pl.Float64).alias("min"),
                pl.col(col).drop_nulls().max().cast(pl.Float64).alias("max"),
                pl.col(col).drop_nulls().median().cast(pl.Float64).alias("median"),
            )
            .row(0, named=True)
        )
        start = int(math.floor(row["min"]))
        stop = int(math.ceil(row["max"]))
        value = int(round(row["median"]))
        return start, stop, value

    def float_slider_stats(df: pl.DataFrame, col: str):
        row = (
            df.select(
                pl.col(col).drop_nulls().min().cast(pl.Float64).alias("min"),
                pl.col(col).drop_nulls().max().cast(pl.Float64).alias("max"),
                pl.col(col).drop_nulls().median().cast(pl.Float64).alias("median"),
            )
            .row(0, named=True)
        )
        start = float(row["min"])
        stop = float(row["max"])
        value = float(row["median"])
        return start, stop, value


    return float_slider_stats, int_slider_stats, mode_option, unique_options


@app.cell(hide_code=True)
def _(
    X_train,
    best_models,
    float_slider_stats,
    int_slider_stats,
    mo,
    mode_option,
    unique_options,
):
    model_dd = mo.ui.dropdown(
        options=list(best_models.keys()),
        value=list(best_models.keys())[0],
        label="Model"
    )

    title_dd = mo.ui.dropdown(
        options=unique_options(X_train, "title"),
        value=mode_option(X_train, "title"),
        label="Title"
    )

    cabinprefix_dd = mo.ui.dropdown(
        options=unique_options(X_train, "cabinprefix"),
        value=mode_option(X_train, "cabinprefix"),
        label="Cabin prefix"
    )

    embark_at_dd = mo.ui.dropdown(
        options=unique_options(X_train, "embark_at"),
        value=mode_option(X_train, "embark_at"),
        label="Embark at"
    )

    pclass_dd = mo.ui.dropdown(
        options=unique_options(X_train, "pclass"),
        value=mode_option(X_train, "pclass"),
        label="Passenger class"
    )

    sex_dd = mo.ui.dropdown(
        options=unique_options(X_train, "sex"),
        value=mode_option(X_train, "sex"),
        label="Sex"
    )

    ticketprefix_dd = mo.ui.dropdown(
        options=unique_options(X_train, "ticketprefix"),
        value=mode_option(X_train, "ticketprefix"),
        label="Ticket prefix"
    )

    _age_start, _age_stop, _age_default = int_slider_stats(X_train, "age")
    _family_start, _family_stop, _family_default = int_slider_stats(X_train, "family_size")
    _fare_start, _fare_stop, _fare_default = float_slider_stats(X_train, "fare")

    age_slider = mo.ui.slider(
        start=_age_start,
        stop=_age_stop,
        value=_age_default,
        step=1,
        label="Age"
    )

    family_slider = mo.ui.slider(
        start=_family_start,
        stop=_family_stop,
        value=_family_default,
        step=1,
        label="Family size"
    )

    fare_slider = mo.ui.slider(
        start=_fare_start,
        stop=_fare_stop,
        value=round(_fare_default, 2),
        step=0.5 if _fare_stop <= 100 else 1.0,
        label="Fare"
    )
    return (
        age_slider,
        cabinprefix_dd,
        embark_at_dd,
        family_slider,
        fare_slider,
        model_dd,
        pclass_dd,
        sex_dd,
        ticketprefix_dd,
        title_dd,
    )


@app.cell(hide_code=True)
def _(
    age_slider,
    cabinprefix_dd,
    embark_at_dd,
    family_slider,
    fare_slider,
    mo,
    model_dd,
    pclass_dd,
    sex_dd,
    ticketprefix_dd,
    title_dd,
):
    ui = mo.vstack(
        [
            mo.md("## Passenger input"),
            mo.hstack([model_dd]),
            mo.hstack([title_dd, sex_dd, pclass_dd]),
            mo.hstack([embark_at_dd, cabinprefix_dd, ticketprefix_dd]),
            mo.hstack([age_slider, family_slider, fare_slider]),
        ]
    )

    ui
    return


@app.cell(hide_code=True)
def _(
    age_slider,
    cabinprefix_dd,
    embark_at_dd,
    family_slider,
    fare_slider,
    one_hot_encode,
    pclass_dd,
    sex_dd,
    ticketprefix_dd,
    title_dd,
):
    input_df = pl.DataFrame(
        {
            "title": [title_dd.value],
            "cabinprefix": [cabinprefix_dd.value],
            "embark_at": [embark_at_dd.value],
            "pclass": pl.Series([pclass_dd.value]).cast(pl.Int8),
            "sex": [sex_dd.value],
            "ticketprefix": [ticketprefix_dd.value],
            "age": pl.Series([age_slider.value]).cast(pl.Float32),
            "family_size": pl.Series([family_slider.value]).cast(pl.Int8),
            "fare": pl.Series([float(fare_slider.value)]).cast(pl.Float32),
        }
    )

    # Transform with your already-fitted One Hot Encoder
    X_new_encoded = one_hot_encode(input_df)
    X_new_encoded
    return (X_new_encoded,)


@app.cell(hide_code=True)
def _(X_new_encoded, best_models, mo, model_dd):
    selected_model_name = model_dd.value
    selected_model = best_models[selected_model_name]

    pred = selected_model.predict(X_new_encoded)[0]

    if hasattr(selected_model, "predict_proba"):
        proba = float(selected_model.predict_proba(X_new_encoded)[0, 1])
    else:
        proba = None

    result_md = (
        f"## Prediction result\n"
        f"- **Model:** `{selected_model_name}`\n"
        f"- **Predicted class:** `{pred}`\n"
    )

    if proba is not None:
        result_md += f"- **Predicted probability (class 1):** `{proba:.2%}`"

    mo.md(result_md)
    return


if __name__ == "__main__":
    app.run()
