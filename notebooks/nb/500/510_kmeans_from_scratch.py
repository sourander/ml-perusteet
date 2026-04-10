import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import random

    import polars as pl
    import altair as alt

    from drawdata import ScatterWidget
    from sklearn.metrics import silhouette_score

    from dataclasses import dataclass

    @dataclass
    class Point:
        coordinates: tuple[float, ...]
        cluster_id: int = -1

    @dataclass
    class Centroid:
        coordinates: tuple[float, ...]
        id: int


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## k-Means from Scratch
    """)
    return


@app.function
def euclidean_distance(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    return sum((pi - qi) ** 2 for pi, qi in zip(p, q)) ** 0.5


@app.class_definition
class kMeans:
    def __init__(self, k: int):
        self.k = k

        self.iter = 0
        self._is_converged = False
        self._converged_reason = ""
        self.k_is_stopped = {i: False for i in range(k)}

        self.centroids: list[Centroid] = []
        self.points: list[Point] = []

    def initialize_centroids(self) -> list[Centroid]:
        k_points = random.sample(self.points, self.k)
        return [Centroid(coordinates=point.coordinates, id=i) for i, point in enumerate(k_points)]

    @staticmethod
    def data_to_points(data: list[tuple[float, ...]]) -> list[Point]:
        return [Point(coordinates=tuple(point)) for point in data]

    @staticmethod
    def mean_coordinates(coordinates: list[tuple[float, ...]]) -> tuple[float, ...]:
        return tuple(sum(x) / len(coordinates) for x in zip(*coordinates))

    def assign_cluster(self, point: Point):
        distances = [euclidean_distance(point.coordinates, centroid.coordinates) for centroid in self.centroids]
        point.cluster_id = distances.index(min(distances))

    def update_centroid(self, centroid: Centroid):
        cluster_points = [point.coordinates for point in self.points if point.cluster_id == centroid.id]

        if not cluster_points:
            self.k_is_stopped[centroid.id] = True
            return

        new_coords = self.mean_coordinates(cluster_points)

        if centroid.coordinates == new_coords:
            self.k_is_stopped[centroid.id] = True
        else:
            self.k_is_stopped[centroid.id] = False
            centroid.coordinates = new_coords

    def fit(self, data: list[tuple[float, ...]], max_iter: int = 100):
        if len(data) <= self.k:
            raise ValueError("Number of clusters k should be less than the number of data points")

        self.points = self.data_to_points(data)
        self.centroids = self.initialize_centroids()

        self.iter = 0

        while True:
            for point in self.points:
                self.assign_cluster(point)

            for centroid in self.centroids:
                self.update_centroid(centroid)

            self.iter += 1

            if self.iter >= max_iter:
                self._is_converged = True
                self._converged_reason = f"Reached max_iter of {max_iter}"

            if all(self.k_is_stopped.values()):
                self._is_converged = True
                self._converged_reason = "All centroids have converged"

            if self._is_converged:
                break

    def predict(self, data: list[tuple[float, ...]]) -> list[int]:
        if not self._is_converged:
            raise ValueError("Model has not been fitted yet")

        points = self.data_to_points(data)
        for point in points:
            self.assign_cluster(point)
        return [point.cluster_id for point in points]

    def inertia(self) -> float:
        total = 0.0
        for point in self.points:
            centroid = self.centroids[point.cluster_id]
            total += euclidean_distance(point.coordinates, centroid.coordinates) ** 2
        return total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw the Dataset
    """)
    return


@app.cell
def _(mo):
    widget = mo.ui.anywidget(ScatterWidget(n_classes=1))
    widget
    return (widget,)


@app.cell
def _(mo, widget):
    mo.stop(not widget.data, "You need to draw some data first. Use the drawdata widget above.")

    data_tuples = [(d["x"], d["y"]) for d in widget.data]
    return (data_tuples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose k and Visualize Clusters
    """)
    return


@app.cell
def _(mo):
    k_slider = mo.ui.slider(1, 15, value=3, step=1, label="k")
    k_slider
    return (k_slider,)


@app.cell
def _(data_tuples, k_slider):
    model = kMeans(k=k_slider.value)
    model.fit(data_tuples)
    return (model,)


@app.cell(hide_code=True)
def _(model):
    points_df = pl.DataFrame({
        "x": [p.coordinates[0] for p in model.points],
        "y": [p.coordinates[1] for p in model.points],
        "cluster": [str(p.cluster_id) for p in model.points],
    })

    centroids_df = pl.DataFrame({
        "x": [c.coordinates[0] for c in model.centroids],
        "y": [c.coordinates[1] for c in model.centroids],
        "cluster": [str(c.id) for c in model.centroids],
    })

    points_chart = (
        alt.Chart(points_df)
        .mark_circle(size=40, opacity=0.85)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("cluster:N"),
            tooltip=[
                alt.Tooltip("x:Q", format=".2f"),
                alt.Tooltip("y:Q", format=".2f"),
                alt.Tooltip("cluster:N"),
            ],
        )
        .properties(width="container", height=400)
    )

    centroids_chart = (
        alt.Chart(centroids_df)
        .mark_point(size=200, shape="cross", filled=True, stroke="black", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("cluster:N"),
            tooltip=[
                alt.Tooltip("x:Q", format=".2f"),
                alt.Tooltip("y:Q", format=".2f"),
                alt.Tooltip("cluster:N"),
            ],
        )
    )

    points_chart + centroids_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Elbow Method
    """)
    return


@app.cell(hide_code=True)
def _(data_tuples):
    max_k = min(15, len(data_tuples) - 1)

    elbow_rows = []
    for _k in range(1, max_k + 1):
        _model = kMeans(k=_k)
        _model.fit(data_tuples)
        elbow_rows.append({"k": _k, "Inertia": _model.inertia()})

    elbow_df = pl.DataFrame(elbow_rows)

    elbow_chart = (
        alt.Chart(elbow_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:O", title="k"),
            y=alt.Y("Inertia:Q", title="Inertia (Sum of Squared Errors)"),
            tooltip=[
                alt.Tooltip("k:O"),
                alt.Tooltip("WCSS:Q", format=".1f"),
            ],
        )
        .properties(width="container", height=300)
    )

    elbow_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Silhouette Analysis
    """)
    return


@app.cell(hide_code=True)
def _(data_tuples):
    max_k_sil = min(15, len(data_tuples) - 1)

    silhouette_rows = []
    for _k in range(2, max_k_sil + 1):
        _model = kMeans(k=_k)
        _model.fit(data_tuples)
        labels = [p.cluster_id for p in _model.points]
        score = silhouette_score(data_tuples, labels)
        silhouette_rows.append({"k": _k, "Silhouette": score})

    silhouette_df = pl.DataFrame(silhouette_rows)

    silhouette_chart = (
        alt.Chart(silhouette_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:O", title="k"),
            y=alt.Y("Silhouette:Q", title="Average Silhouette Score"),
            tooltip=[
                alt.Tooltip("k:O"),
                alt.Tooltip("Silhouette:Q", format=".3f"),
            ],
        )
        .properties(width="container", height=300)
    )

    silhouette_chart
    return


if __name__ == "__main__":
    app.run()
