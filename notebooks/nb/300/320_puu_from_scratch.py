import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path
    from math import log2
    from dataclasses import dataclass

    @dataclass
    class IGScore:
        column_index: int
        column_type: str
        score: float
        split_point: float | None


    @dataclass
    class Node:
        n_samples: int
        depth: int


    @dataclass
    class Leaf(Node):
        label: int
        exit_reason: str


    @dataclass
    class Decision(Node):
        count: int
        ig: IGScore
        left: Node
        right: Node


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decision Tree from Scratch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Familiar helper functions

    You have seen these in the lesson material. Check the **Entropia** and the **Puu** lesson.
    """)
    return


@app.cell
def _():
    def entropy(X:list[int]) -> float:
        H_val = -sum([p * log2(p) for p in X if p > 0])
        return H_val


    def class_probabilities(column_values: list[int]) -> tuple[float, float]:
        n = len(column_values)
        if n == 0:
            return (0.0, 0.0)
        zeros = column_values.count(0)
        ones = column_values.count(1)
        return (zeros / n, ones / n)

    return class_probabilities, entropy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computing Information Gain
    """)
    return


@app.cell
def _(class_probabilities, entropy):
    def calculate_information_gain(left: list[int], right: list[int], H_before: float) -> float:

        # Compute the entropy of the two partitions
        H_left = entropy(class_probabilities(left))
        H_right = entropy(class_probabilities(right))

        # Compute the entropy after the split
        n = len(left) + len(right)

        # Weighted by the number of elements in each partition
        q_left = len(left) / n
        q_right = len(right) / n
        H_after = q_left * H_left + q_right * H_right

        # Compute the information gain
        IG = H_before - H_after
        return IG

    return (calculate_information_gain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finding the optimal split
    """)
    return


@app.cell
def _(calculate_information_gain, class_probabilities, entropy):
    def find_optimal_split_point(data: list[tuple], i: int) -> float | None:
        H_before = entropy(class_probabilities([row[-1] for row in data]))
        column_values = sorted([row[i] for row in data])  # Sort the values

        # Init
        best_ig = float("-inf")
        best_split_point = None

        for idx in range(len(column_values) - 1):
            # Try every possible average of two adjacent values
            # As a candidate split point
            split_point = (column_values[idx] + column_values[idx + 1]) / 2
            left = [row[-1] for row in data if row[i] <= split_point]
            right = [row[-1] for row in data if row[i] > split_point]

            # Calculate information gain for this split
            ig = calculate_information_gain(left, right, H_before)

            # Update the best split point if information gain is better
            if ig > best_ig:
                best_ig = ig
                best_split_point = split_point

        return best_split_point

    return (find_optimal_split_point,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Column Information Gain
    """)
    return


@app.cell
def _(
    calculate_information_gain,
    class_probabilities,
    entropy,
    find_optimal_split_point,
):
    def column_information_gain(data: list[tuple], i: int) -> IGScore:

        if len(data) == 0:
            raise ValueError("Data must not be empty")

        # Initialize a variable to store the best split value
        best_split_point = 0.5

        # Compute the entropy of the entire dataset
        # This is the entropy before the split
        H_before = entropy(class_probabilities([row[-1] for row in data]))

        # Check that split index column values are binary
        binary_column = all([r[i] in (0, 1) for r in data])

        if binary_column:
            # Partition the data into left and right partitions
            # Based on the binary class label (e.g. there is a shower)
            left = [row[-1] for row in data if not row[i]]
            right = [row[-1] for row in data if row[i]]

        else:
            best_split_point = find_optimal_split_point(data, i)

            # Use the best split point to partition the data
            left = [row[-1] for row in data if row[i] <= best_split_point]
            right = [row[-1] for row in data if row[i] > best_split_point]

        ig = calculate_information_gain(left, right, H_before)
        return IGScore(
            column_index=i,
            column_type="binary" if binary_column else "continuous",
            score=ig,
            split_point=best_split_point,
        )

    def find_max_column_information_gain(data: list[tuple], verbose: bool = False) -> IGScore | None:
        # Find out the best column to split on
        best_ig: IGScore | None = None

        # We assume that the last column is the class label
        N_FEATURES = len(data[0]) - 1

        for i in range(N_FEATURES):
            this_ig: IGScore = column_information_gain(data, i)

            if verbose:
                print(f"[find_max_column_information_gain] IG for column {i}: {this_ig}")

            if best_ig is None or this_ig.score > best_ig.score:
                best_ig = this_ig

        return best_ig

    return (find_max_column_information_gain,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splitter
    """)
    return


@app.function
def split_data(data: list[tuple], ig: IGScore) -> tuple[list[tuple], list[tuple]]:

    # Produce the split based on binary or continuous column
    if ig.column_type == "binary":
        left = [row for row in data if not row[ig.column_index]]
        right = [row for row in data if row[ig.column_index]]
    elif ig.column_type == "continuous":
        left = [row for row in data if row[ig.column_index] <= ig.split_point]
        right = [row for row in data if row[ig.column_index] > ig.split_point]
    else:
        raise ValueError(f"Unknown column type {ig.column_type}")

    return left, right


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a Tree
    """)
    return


@app.cell
def _(find_max_column_information_gain):
    def build_tree(data: list[tuple], depth: int = 0, **kwargs) -> Node:

        def majority_class(reason: str) -> Leaf:
            """
            A nested helper function to create a Leaf node
            on early exit with a given reason.
            """
            return Leaf(
                n_samples=len(data),
                depth=depth,
                label=max(set([row[-1] for row in data]), key=[row[-1] for row in data].count),
                exit_reason=reason
            )

        # Default values
        max_depth = kwargs.get("max_depth", 5)
        min_ig = kwargs.get("min_ig", 0.01)
        min_samples_split = kwargs.get("min_samples_split", 2)
        verbose = kwargs.get("verbose", False)

        if verbose:
            print(
                f"[build_tree] Building a Decision or Leaf at depth {depth} with {len(data)} samples"
            )

        # Stop condition: we have reached a uniform class
        if len(set([row[-1] for row in data])) == 1:
            return majority_class("Uniform class")

        # Early stopping if we reach the maximum depth
        # Predict the most common class
        if depth >= max_depth:
            print(f"[build_tree] Early stopping at depth {depth}")
            return majority_class("Max depth")

        # Stop if the number of samples is below the minimum required to split
        if len(data) < min_samples_split:
            return majority_class("Min samples split")

        # Find out the best column to split on
        ig = find_max_column_information_gain(data, verbose=verbose)

        if ig.score < min_ig:
            return majority_class("Min IG")

        # Split the data based on the best column
        left, right = split_data(data, ig)

        # Create a new node
        node = Decision(
            n_samples=len(data),
            depth=depth,
            count=len(data),
            ig=ig,
            left=build_tree(left, depth + 1, **kwargs),
            right=build_tree(right, depth + 1, **kwargs),
        )

        return node

    return (build_tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Functions you should use
    """)
    return


@app.cell
def _():
    def visualize_tree(node: Node, indent: int = 0) -> None:
        col_names = ["im_well_rested", "dst_has_shower", "required_speed", "go_by_car"]

        if isinstance(node, Leaf):
            print("  " * indent + str(node))
        elif isinstance(node, Decision):
            print(
                "  " * indent + f"Decision ({node.depth}, n={node.n_samples}): "
                f"Split on {col_names[node.ig.column_index]} ({node.ig.split_point})"
            )
            visualize_tree(node.left, indent + 1)
            visualize_tree(node.right, indent + 1)


    def predict(node: Node, input_values: tuple) -> int:
        while True:
            # If we have reached a leaf, return the label
            if isinstance(node, Leaf):
                return node.label

            if not isinstance(node, Decision):
                raise ValueError(f"Unknown node type {type(node)}")

            if node.ig.column_type == "binary":
                if input_values[node.ig.column_index] == 0:
                    node = node.left
                else:
                    node = node.right
            else:
                if input_values[node.ig.column_index] <= node.ig.split_point:
                    node = node.left
                else:
                    node = node.right

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

    return predict, read_jsonl, visualize_tree


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive Part

    To start, you should unzip the data by running...

    ```bash
    cd notebooks
    unzip gitlfs-store/tree-biking.zip
    ```

    You should end up having these files...

    ```
    data/bike_or_car
    ├── 100_test.jsonl
    ├── 16_row_sample.jsonl
    └── 293_train.jsonl
    ```

    The first number is the number of lines. The `16_row_sample.jsonl` contains 16 rows. This is to make printing to be less verbose when you want to find out how the algorithm works. When you do know how it works, you can use the `293_train.jsonl` to train the model, and `100_test.jsonl` to evaluate your model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16 rows
    """)
    return


@app.cell
def _(build_tree, read_jsonl):
    data_16 = read_jsonl(Path("data/bike_or_car/16_row_sample.jsonl"))

    tree_sample = build_tree(data_16, max_depth=3, verbose=True)
    return (tree_sample,)


@app.cell
def _(tree_sample, visualize_tree):
    visualize_tree(tree_sample)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BIG DATA! 293 + 100 rows!
    """)
    return


@app.cell
def _(build_tree, read_jsonl):
    # Run against the test data
    print("\n==== Test with larger dataset begins ====\n")
    data_train = read_jsonl(Path("data/bike_or_car/293_train.jsonl"))
    data_test = read_jsonl(Path("data/bike_or_car/100_test.jsonl"))
    tree_293 = build_tree(data_train, max_depth=5, verbose=True)
    return data_test, tree_293


@app.cell
def _(tree_293, visualize_tree):
    visualize_tree(tree_293)
    return


@app.cell
def _(data_test, predict, tree_293):
    y = [row[-1] for row in data_test]
    y_hat = [predict(tree_293, row) for row in data_test]
    false_positives = sum([a == 0 and b == 1 for a, b in zip(y, y_hat)])
    true_positives = sum([a == 1 and b == 1 for a, b in zip(y, y_hat)])
    false_negatives = sum([a == 1 and b == 0 for a, b in zip(y, y_hat)])
    true_negatives = sum([a == 0 and b == 0 for a, b in zip(y, y_hat)])
    print("Confusion matrix:")
    print(f"TP: {true_positives}", f"FP: {false_positives}")
    print(f"FN: {false_negatives}", f"TN: {true_negatives}")
    print(f"Accuracy: {(true_positives + true_negatives) / len(y)}")
    return


if __name__ == "__main__":
    app.run()
