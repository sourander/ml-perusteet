import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import numpy as np
    import altair as alt

    from drawdata import ScatterWidget

    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, TSNE
    from sklearn.datasets import load_digits


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Draw Data
    """)
    return


@app.cell
def _(mo):
    widget = mo.ui.anywidget(ScatterWidget())
    widget
    return (widget,)


@app.cell
def _(widget):
    df_raw = widget.data_as_polars
    df_raw
    return (df_raw,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2D

    ## 2D Helper Functions
    """)
    return


@app.cell(hide_code=True)
def _():
    def xy_as_numpy(df: pl.DataFrame, x_col: str = "x", y_col: str = "y") -> np.ndarray:
        """Extract two numeric columns from a Polars DataFrame as an (n, 2) NumPy array."""
        return df.select([x_col, y_col]).to_numpy()


    def add_embedding_columns(
        df: pl.DataFrame,
        embedding: np.ndarray,
        x_new: str = "x_proj",
        y_new: str = "y_proj",
    ) -> pl.DataFrame:
        """Return a new Polars DataFrame with embedding columns appended."""
        return df.with_columns(
            pl.Series(x_new, embedding[:, 0]),
            pl.Series(y_new, embedding[:, 1]),
        )


    def rotate_2d(X: np.ndarray, angle_deg: float) -> np.ndarray:
        """Pure geometric coordinate transform in 2D."""
        theta = np.deg2rad(angle_deg)
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
            ]
        )
        return X @ R.T


    def center_2d(X: np.ndarray) -> np.ndarray:
        """Center points around the origin."""
        return X - X.mean(axis=0, keepdims=True)


    def transform_2d(
        X: np.ndarray,
        method: str,
        random_state: int = 42,
        tsne_perplexity: float = 30.0,
        tsne_init: str = "pca",
    ) -> np.ndarray:
        """
        Apply a chosen 2D -> 2D transform.

        Methods:
          - identity
          - center
          - pca
          - mds
          - tsne
        """
        if method == "identity":
            Z = X.copy()

        elif method == "center":
            Z = center_2d(X)

        elif method == "pca":
            # 2D -> 2D PCA is a change of basis (rotation/reflection + centering)
            model = PCA(n_components=2)
            Z = model.fit_transform(X)

        elif method == "mds":
            model = MDS(
                n_components=2,
                metric_mds=True,
                metric="euclidean",
                init="random",
                random_state=random_state,
                n_init=1,
                max_iter=300,
            )
            Z = model.fit_transform(X)

        elif method == "tsne":
            # Perplexity must be < number of samples
            perplexity = min(float(tsne_perplexity), max(2.0, X.shape[0] - 1.0))

            model = TSNE(
                n_components=2,
                perplexity=perplexity,
                init=tsne_init,
                learning_rate="auto",
                random_state=random_state,
            )
            Z = model.fit_transform(X)

        else:
            raise ValueError(f"Unknown method: {method}")

        return Z


    def scatter_chart(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        *,
        title: str,
        width: int = 320,
        height: int = 320,
    ):
        """
        Altair scatter plot using Polars data without Pandas.
        Assumes the DataFrame contains columns: color, label.
        """
        return (
            alt.Chart(df.to_arrow())
            .mark_circle(size=40, opacity=0.85)
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_col, scale=alt.Scale(zero=False)),
                y=alt.Y(f"{y_col}:Q", title=y_col, scale=alt.Scale(zero=False)),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip(f"{x_col}:Q", format=".2f"),
                    alt.Tooltip(f"{y_col}:Q", format=".2f"),
                    alt.Tooltip("label:N"),
                    alt.Tooltip("batch:Q"),
                ],
            )
            .properties(title=title, width=width, height=height)
        )


    return add_embedding_columns, scatter_chart, transform_2d, xy_as_numpy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D to 2D DataFrame
    """)
    return


@app.cell(hide_code=True)
def _(
    add_embedding_columns,
    df_raw,
    method_ui,
    perplexity_ui,
    transform_2d,
    xy_as_numpy,
):
    X = xy_as_numpy(df_raw, x_col="x", y_col="y")

    Z = transform_2d(
        X,
        method=method_ui.value,
        tsne_perplexity=float(perplexity_ui.value),
    )

    df_2d = add_embedding_columns(df_raw, Z, x_new="x_proj", y_new="y_proj")
    return (df_2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D to 2D Chart
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    method_ui = mo.ui.dropdown(
        options={
            "Identity": "identity",
            "Center to origin": "center",
            "PCA (2 → 2)": "pca",
            "MDS (2 → 2)": "mds",
            "t-SNE (2 → 2)": "tsne",
        },
        value="PCA (2 → 2)",
        label="2D → 2D method",
    )

    perplexity_ui = mo.ui.slider(
        start=2,
        stop=80,
        step=1,
        value=30,
        label="t-SNE perplexity",
    )

    mo.vstack(
        [
            mo.hstack([method_ui, perplexity_ui]),
        ]
    )
    return method_ui, perplexity_ui


@app.cell(hide_code=True)
def _(df_2d, df_raw, method_ui, scatter_chart):
    original_chart = scatter_chart(
        df_raw,
        x_col="x",
        y_col="y",
        title="Original 2D data",
    )

    transformed_chart = scatter_chart(
        df_2d,
        x_col="x_proj",
        y_col="y_proj",
        title=f"Transformed: {method_ui.value}",
    )

    original_chart | transformed_chart
    return (original_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1D

    ## 1D Helper Functions
    """)
    return


@app.cell(hide_code=True)
def _():
    def add_projection_column(
        df: pl.DataFrame,
        projection: np.ndarray,
        col_name: str = "z_1d",
    ) -> pl.DataFrame:
        """Return a new Polars DataFrame with a 1D projection column appended."""
        if projection.ndim == 2 and projection.shape[1] == 1:
            values = projection[:, 0]
        elif projection.ndim == 1:
            values = projection
        else:
            raise ValueError(
                f"Projection must have shape (n,) or (n, 1), got {projection.shape}"
            )

        return df.with_columns(pl.Series(col_name, values))


    def transform_1d(
        X: np.ndarray,
        method: str,
        *,
        random_state: int = 42,
        tsne_perplexity: float = 30.0,
    ) -> np.ndarray:
        """
        Apply a chosen 2D -> 1D transform.

        Methods:
          - keep_first
          - keep_second
          - pca
          - mds
          - tsne

        Returns
        -------
        np.ndarray of shape (n_samples, 1)
        """
        if method == "keep_first":
            Z = X[:, [0]]

        elif method == "keep_second":
            Z = X[:, [1]]

        elif method == "pca":
            model = PCA(n_components=1)
            Z = model.fit_transform(X)

        elif method == "mds":
            # Workaround for current sklearn bug:
            # init="classical_mds" can return 2 components even when n_components=1.
            model = MDS(
                n_components=1,
                metric_mds=True,
                metric="euclidean",
                init="random",
                n_init=4,
                random_state=random_state,
                max_iter=300,
            )
            Z = model.fit_transform(X)


        elif method == "tsne":
            perplexity = min(float(tsne_perplexity), max(2.0, X.shape[0] - 1.0))

            model = TSNE(
                n_components=1,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            )
            Z = model.fit_transform(X)

        else:
            raise ValueError(f"Unknown 2D -> 1D method: {method}")

        return Z

    def chart_1d_projection(
        df: pl.DataFrame,
        z_col: str = "z_1d",
        *,
        title: str = "1D projection",
        width: int = 700,
        height: int = 140,
        separate_by_label: bool = False,
    ):
        """
        Create an Altair chart for a 1D projection stored in `z_col`.

        Parameters
        ----------
        df
            Polars DataFrame containing at least:
            - z_col
            - color
            - label
            - batch
        z_col
            Name of the 1D coordinate column.
        title
            Chart title.
        width, height
            Chart dimensions.
        separate_by_label
            If False, draw a pure 1D rug/strip plot on one line.
            If True, draw one horizontal row per label.
        """
        source = df.to_arrow()

        if separate_by_label:
            return (
                alt.Chart(source)
                .mark_circle(size=42, opacity=0.80)
                .encode(
                    x=alt.X(
                        f"{z_col}:Q",
                        title="1D coordinate",
                        scale=alt.Scale(zero=False),
                    ),
                    y=alt.Y(
                        "label:N",
                        title="label",
                    ),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip(f"{z_col}:Q", format=".2f"),
                        alt.Tooltip("label:N"),
                        alt.Tooltip("batch:Q"),
                    ],
                )
                .properties(title=title, width=width, height=height)
            )

        return (
            alt.Chart(source)
            .mark_tick(thickness=2, size=18, opacity=0.85)
            .encode(
                x=alt.X(
                    f"{z_col}:Q",
                    title="1D coordinate",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip(f"{z_col}:Q", format=".2f"),
                    alt.Tooltip("label:N"),
                    alt.Tooltip("batch:Q"),
                ],
            )
            .properties(title=title, width=width, height=height)
        )

    return add_projection_column, chart_1d_projection, transform_1d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1D to 1D DataFrame
    """)
    return


@app.cell(hide_code=True)
def _(
    add_projection_column,
    df_raw,
    method_ui_1d,
    perplexity_ui_1d,
    transform_1d,
    xy_as_numpy,
):
    X1d = xy_as_numpy(df_raw, x_col="x", y_col="y")

    Z1d = transform_1d(
        X1d,
        method=method_ui_1d.value,
        tsne_perplexity=float(perplexity_ui_1d.value),
    )

    df_1d = add_projection_column(df_raw, Z1d, col_name="z_1d")
    return (df_1d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1D to 1D Chart
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    method_ui_1d = mo.ui.dropdown(
        options={
            "Keep 1st": "keep_first",
            "Keep 2nd": "keep_second",
            "PCA (2 → 1)": "pca",
            "MDS (2 → 1)": "mds",
            "t-SNE (2 → 1)": "tsne",
        },
        value="PCA (2 → 1)",
        label="2D → 1D method",
    )

    perplexity_ui_1d = mo.ui.slider(
        start=2,
        stop=80,
        step=1,
        value=30,
        label="t-SNE perplexity",
    )

    mo.vstack(
        [
            mo.hstack([method_ui_1d, perplexity_ui_1d]),
        ]
    )
    return method_ui_1d, perplexity_ui_1d


@app.cell
def _(chart_1d_projection, df_1d, method_ui_1d, original_chart):
    chart_1d = chart_1d_projection(
        df_1d,
        z_col="z_1d",
        title=f"1D projection: {method_ui_1d.value}",
        separate_by_label=False,
    )

    original_chart | chart_1d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MNIST
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MNIST Helper Functions
    """)
    return


@app.cell(hide_code=True)
def _():
    def embed_digits_2d(
        X: np.ndarray,
        method: str,
        *,
        random_state: int = 42,
        tsne_perplexity: float = 30.0,
    ) -> np.ndarray:
        """
        Methods:
          - pca
          - mds
          - tsne
        """
        if method == "pca":
            Z = PCA(n_components=2).fit_transform(X)

        elif method == "mds":
            Z = MDS(
                n_components=2,
                metric_mds=True,
                metric="euclidean",
                init="random",
                random_state=random_state,
                n_init=1,
                max_iter=300,
            ).fit_transform(X)

        elif method == "tsne":
            perplexity = min(float(tsne_perplexity), max(2.0, X.shape[0] - 1.0))
            Z = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            ).fit_transform(X)

        else:
            raise ValueError(f"Unknown digits embedding method: {method}")

        return Z

    def digits_scatter_chart(
        df: pl.DataFrame,
        x_col: str = "x_proj",
        y_col: str = "y_proj",
        *,
        title: str,
        width: int = 600,
        height: int = 420,
    ):
        digit_domain = [str(i) for i in range(10)]
        digit_range = [
            "#1f77b4",  # 0
            "#ff7f0e",  # 1
            "#2ca02c",  # 2
            "#d62728",  # 3
            "#9467bd",  # 4
            "#8c564b",  # 5
            "#e377c2",  # 6
            "#7f7f7f",  # 7
            "#bcbd22",  # 8
            "#17becf",  # 9
        ]

        return (
            alt.Chart(df.to_arrow())
            .mark_circle(size=55, opacity=0.82)
            .encode(
                x=alt.X(
                    f"{x_col}:Q",
                    title="1st dim",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(
                    f"{y_col}:Q",
                    title="2nd dim",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "label:N",
                    title="Digit",
                    scale=alt.Scale(domain=digit_domain, range=digit_range),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Digit"),
                    alt.Tooltip("sample_id:Q", title="Sample"),
                    alt.Tooltip(f"{x_col}:Q", title="x", format=".2f"),
                    alt.Tooltip(f"{y_col}:Q", title="y", format=".2f"),
                ],
            )
            .properties(
                title=title,
                width=width,
                height=height,
            )
        )

    return digits_scatter_chart, embed_digits_2d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MNIST to 2D DataFrame
    """)
    return


@app.cell
def _(
    add_embedding_columns,
    embed_digits_2d,
    mnist_method_ui,
    mnist_n_ui,
    mnist_perplexity_ui,
):
    X_digits, y_digits = load_digits(return_X_y=True)

    n_digits = int(mnist_n_ui.value)
    X_digits_small = X_digits[:n_digits]
    y_digits_small = y_digits[:n_digits]

    Z_digits = embed_digits_2d(
        X_digits_small,
        method=mnist_method_ui.value,
        tsne_perplexity=float(mnist_perplexity_ui.value),
    )

    df_digits = pl.DataFrame(
        {
            "sample_id": np.arange(n_digits),
            "label": y_digits_small.astype(str),
        }
    )

    df_digits = add_embedding_columns(
        df_digits,
        Z_digits,
        x_new="x_proj",
        y_new="y_proj",
    )

    df_digits
    return X_digits_small, df_digits, n_digits


@app.cell(hide_code=True)
def _(mo):
    mnist_method_ui = mo.ui.dropdown(
        options={
            "PCA (64 → 2)": "pca",
            "MDS (64 → 2)": "mds",
            "t-SNE (64 → 2)": "tsne",
        },
        value="PCA (64 → 2)",
        label="Digits 2D -menetelmä",
    )

    mnist_n_ui = mo.ui.slider(
        start=50,
        stop=500,
        step=50,
        value=200,
        label="n digits to keep",
    )

    mnist_perplexity_ui = mo.ui.slider(
        start=5,
        stop=50,
        step=1,
        value=30,
        label="t-SNE perplexity",
    )

    mo.vstack(
        [
            mo.hstack([mnist_method_ui, mnist_n_ui, mnist_perplexity_ui]),
        ]
    )
    return mnist_method_ui, mnist_n_ui, mnist_perplexity_ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MNIST 2D Chart
    """)
    return


@app.cell
def _(df_digits, digits_scatter_chart, mnist_method_ui, n_digits):
    digits_chart = digits_scatter_chart(
        df_digits,
        x_col="x_proj",
        y_col="y_proj",
        title=f"MNIST digits in 2D ({mnist_method_ui.value}, n={n_digits})",
    )

    digits_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trying to Interpret PCA

    The principal components are really hard to interpret. Since each component is just a set of weights for the pixels, we can visualize them as heatmaps. This shows which parts of the image contribute positively or negatively to the component. Even so, it is difficult to explain PC1 or PC2 in plain language: they are mathematical patterns in the data, not clear human-readable features.
    """)
    return


@app.cell
def _(X_digits_small):
    pca_digits = PCA(n_components=2)
    Z_digits_pca = pca_digits.fit_transform(X_digits_small)
    pc1_img = pca_digits.components_[0].reshape(8, 8)
    pc2_img = pca_digits.components_[1].reshape(8, 8)
    return pc1_img, pc2_img


@app.cell(hide_code=True)
def _(pc1_img, pc2_img):
    m_1 = np.abs(pc1_img).max()
    df_pc1 = pl.DataFrame(pc1_img).with_row_index("row").unpivot(
        index="row", variable_name="col", value_name="value"
    )
    m_2 = np.abs(pc2_img).max()
    df_pc2 = pl.DataFrame(pc2_img).with_row_index("row").unpivot(
        index="row", variable_name="col", value_name="value"
    )

    _chart_pc1 = alt.Chart(df_pc1).mark_rect().encode(
        x="col:O",
        y=alt.Y("row:O", sort="descending"),  # row 0 at top
        color=alt.Color("value:Q", scale=alt.Scale(domain=[-m_1, m_1], scheme="redblue")),
        tooltip=["row:Q", "col:O", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=320, height=320)

    _chart_pc2 = alt.Chart(df_pc2).mark_rect().encode(
        x="col:O",
        y=alt.Y("row:O", sort="descending"),  # row 0 at top
        color=alt.Color("value:Q", scale=alt.Scale(domain=[-m_2, m_2], scheme="redblue")),
        tooltip=["row:Q", "col:O", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=320, height=320)

    _chart_pc1 | _chart_pc2
    return


if __name__ == "__main__":
    app.run()
