import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")

with app.setup:
    import polars as pl
    import altair as alt
    import numpy as np

    from drawdata import ScatterWidget

    from collections import defaultdict
    from dataclasses import dataclass

    @dataclass
    class Point:
        x: float
        y: float
        color: str
        label: str | None

    @dataclass
    class Neighbor:
        distance: float
        label: str|int


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## k-NN from Scratch
    """)
    return


@app.function
def euclidean_distance(p1: Point, p2: Point) -> float:
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


@app.class_definition
class kNN:
    def __init__(self, k: int):
        self.k = k
        self.training_points: list[Point] = []

    def fit(self, points: list[Point]):
        self.training_points = points

    @staticmethod
    def majority_vote(neighbors: list[Neighbor]) -> str:
        votes = defaultdict(int)
        for neighbor in neighbors:
            votes[neighbor.label] += 1
        return max(votes, key=votes.get)

    def get_k_nearest(self, query: Point) -> list[Point]:
        return sorted(
            self.training_points, key=lambda p: euclidean_distance(query, p)
        )[: self.k]

    def predict(self, query: Point) -> str:
        neighbors = [
            Neighbor(distance=euclidean_distance(query, p), label=p.label)
            for p in self.training_points
        ]
        neighbors.sort(key=lambda n: n.distance)
        return self.majority_vote(neighbors[: self.k])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw the Dataset
    """)
    return


@app.cell
def _(mo):
    widget = mo.ui.anywidget(ScatterWidget())
    widget
    return (widget,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train the Model
    """)
    return


@app.cell
def _(mo, widget):
    mo.stop(not widget.data, "You need to draw some data first. Use the drawdata widget above.")

    training_points = [
        Point(x=d["x"], y=d["y"], color=d["color"], label=d["label"])
        for d in widget.data
    ]

    min_x = int(min(training_points, key=lambda p: p.x).x)
    min_y = int(min(training_points, key=lambda p: p.y).y)
    max_x = int(max(training_points, key=lambda p: p.x).x)
    max_y = int(max(training_points, key=lambda p: p.y).y)    
    return max_x, max_y, min_x, min_y, training_points


@app.cell
def _(k_slider, training_points):
    knn = kNN(k=k_slider.value)
    knn.fit(training_points)
    return (knn,)


@app.cell
def _(knn, new_dot_x_slider, new_dot_y_slider, training_points):
    label_to_color = {p.label: p.color for p in training_points}
    query = Point(x=new_dot_x_slider.value, y=new_dot_y_slider.value, color=None, label=None)
    predicted_label = knn.predict(query)
    predicted_color = label_to_color.get(predicted_label, "#888888")
    k_nearest_points = knn.get_k_nearest(query)
    return k_nearest_points, predicted_color, predicted_label


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict a New Point
    """)
    return


@app.cell
def _(max_x, max_y, min_x, min_y, mo, training_points):
    k_slider = mo.ui.slider(1, min(15, len(training_points)), value=1, step=2, label="k")

    new_dot_x_slider = mo.ui.slider(start=min_x, stop=max_x, step=2, label="x", value=(max_x + min_x) // 2)
    new_dot_y_slider = mo.ui.slider(start=min_y, stop=max_y, step=2, label="y", value=(max_y + min_y) // 2)
    new_dot_x_slider, new_dot_y_slider, k_slider

    mo.hstack([new_dot_x_slider, new_dot_y_slider, k_slider])
    return k_slider, new_dot_x_slider, new_dot_y_slider


@app.cell(hide_code=True)
def _(
    k_nearest_points,
    new_dot_x_slider,
    new_dot_y_slider,
    predicted_color,
    predicted_label,
    widget,
):
    df = widget.data_as_polars

    training_chart = (
        alt.Chart(df)
        .mark_circle(size=40, opacity=0.85)
        .encode(
            x=alt.X("x:Q", title="x", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", title="y", scale=alt.Scale(zero=False)),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("x:Q", format=".2f"),
                alt.Tooltip("y:Q", format=".2f"),
                alt.Tooltip("label:N"),
                alt.Tooltip("batch:Q"),
            ],
        )
        .properties(width="container", height=400)
    )

    predicted_df = pl.DataFrame({
        "x": [float(new_dot_x_slider.value)],
        "y": [float(new_dot_y_slider.value)],
        "color": [predicted_color],
        "label": [predicted_label],
    })

    lines_rows = []
    for i, neighbor in enumerate(k_nearest_points):
        lines_rows.append({"group": i, "x": float(new_dot_x_slider.value), "y": float(new_dot_y_slider.value)})
        lines_rows.append({"group": i, "x": neighbor.x, "y": neighbor.y})

    lines_df = pl.DataFrame(lines_rows)
    lines_chart = (
        alt.Chart(lines_df)
        .mark_line(color="gray", strokeDash=[4, 2], opacity=0.6)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", scale=alt.Scale(zero=False)),
            detail="group:N",
        )
    )

    predicted_chart = (
        alt.Chart(predicted_df)
        .mark_point(size=200, shape="cross", filled=True, stroke="black", strokeWidth=1.5)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("y:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("x:Q", format=".2f"),
                alt.Tooltip("y:Q", format=".2f"),
                alt.Tooltip("label:N"),
            ],
        )
    )

    chart = training_chart + lines_chart + predicted_chart
    chart
    return


if __name__ == "__main__":
    app.run()
