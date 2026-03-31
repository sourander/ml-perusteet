import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import random
    import json
    import sklearn.tree

    from pathlib import Path


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Forest from Scratch

    In my previous course iteration, the DIY Decision Tree was used as a part of the forest. The current solution is to use premade Decision Tree and only do the Random Forest part ourselves. However, nothing is stopping you from building a trees that use the `320_puu_from_scratch.py` code.

    ## Sample with replacement
    """)
    return


@app.function
def sample_with_replacement(data):
    """Alias:
    Bagging with bootstrap
    """
    return [random.choice(data) for _ in range(len(data))]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample without replacement
    """)
    return


@app.function
def sample_without_replacement(data, n):
    """Alias:
    Bagging with a random subset
    """
    random.shuffle(data)
    return data[:n]


@app.function
def read_jsonl(file_path: Path) -> list[tuple]:
    # Read
    contents = file_path.read_text(encoding="utf-8")

    # Accumulate
    data = []
    for line in contents.splitlines():
        d = json.loads(line)
        row = (
            d["im_well_rested"],
            d["dst_has_shower"],
            float(d["required_speed"]),
            d["go_by_car"],
        )
        data.append(row)
    return data


@app.class_definition
class RandomForest:
    def __init__(self, num_trees, verbose=False):
        self.num_trees = num_trees
        self.trees: list[sklearn.tree.DecisionTreeClassifier] = []
        self.verbose = verbose

    def train(self, data: list[tuple], replacement=True):

        subsets = []
        if replacement:
            subsets = [sample_with_replacement(data) for _ in range(self.num_trees)]
        else:
            sample_size = len(data) // self.num_trees
            subsets = [
                sample_without_replacement(data, sample_size)
                for _ in range(self.num_trees)
            ]

        for subset in subsets:

            if self.verbose:
                percentage = sum([x[-1] for x in subset]) / len(subset) * 100
                print(
                    f"Building tree with {percentage:.1f} % positive samples (n rows {len(subset)})."
                )

            X = [list(row[:-1]) for row in subset]
            y = [row[-1] for row in subset]
            tree = sklearn.tree.DecisionTreeClassifier()
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, sample):
        labels = [tree.predict([list(sample[:-1])])[0] for tree in self.trees]

        if self.verbose:
            votes = f"({','.join(str(l) for l in labels)})"
            status = "unanimous" if len(set(labels)) == 1 else "CONFLICTING"
            print(f"Votes: {votes} -> {status}")

        return max(set(labels), key=labels.count)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _():
    data_train = read_jsonl(Path("data/bike_or_car/293_train.jsonl"))
    data_test = read_jsonl(Path("data/bike_or_car/100_test.jsonl"))
    return data_test, data_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train
    """)
    return


@app.cell
def _(data_train):
    random_forest = RandomForest(num_trees=5, verbose=True)
    random_forest.train(data_train)
    return (random_forest,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predict
    """)
    return


@app.cell
def _(data_test, random_forest):
    y = [row[-1] for row in data_test]
    y_hat = [random_forest.predict(row) for row in data_test]
    false_positives = sum([a == 0 and b == 1 for a, b in zip(y, y_hat)])
    true_positives = sum([a == 1 and b == 1 for a, b in zip(y, y_hat)])
    false_negatives = sum([a == 1 and b == 0 for a, b in zip(y, y_hat)])
    true_negatives = sum([a == 0 and b == 0 for a, b in zip(y, y_hat)])
    print("Confusion matrix:")
    print(f"TP: {true_positives}", f"FP: {false_positives}")
    print(f"FN: {false_negatives}", f"TN: {true_negatives}")
    print(f"Accuracy: {(true_positives + true_negatives) / len(y)}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
