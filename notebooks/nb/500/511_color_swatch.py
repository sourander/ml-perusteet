import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import cv2
    import numpy as np
    import plotly.graph_objects as go

    from sklearn.cluster import KMeans


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Color Swatch

    The goal of this Notebook is to produce a color swatch of any given color photograph. The user should be able to select the $k$ for how many paint buckets one is willing to buy.

    ## Load and Convert Data

    Load the image, convert it to RGB, HSV and LAB.
    """)
    return


@app.cell
def _(mo):
    img = cv2.imread("data/photos/510_task_kmeans_fruit.jpg")

    # OpenCV default to BGR channel order, Pyplot and Marimo assume RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Resize to make the calculations faster
    img_rgb = cv2.resize(img_rgb, (200, 200))
    img_hsv = cv2.resize(img_hsv, (200, 200))
    img_lab = cv2.resize(img_lab, (200, 200))

    mo.image(img_rgb, width=400)
    return img_hsv, img_lab, img_rgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flatten

    Make the dataset to be one pixel per row. The pixels we are clustering are individual pixels: we do not need xy-coordinates. This will produce $40\,000$ rows and $3$ columns. The colums are either RGB, HSV, LAB or whatever you make them be. Essentially they could even be some PCA components.
    """)
    return


@app.cell
def _(img_hsv, img_lab, img_rgb):
    X_rgb = img_rgb.reshape((-1, 3))
    X_hsv = img_hsv.reshape((-1, 3))
    X_lab = img_lab.reshape((-1, 3))

    print("[INFO] Check the maximum values of all columns to get a sense of their range")
    print("  RGB(max): ", X_rgb.max(axis=0))
    print("  HSV(max): ", X_hsv.max(axis=0))
    print("  LAB(max): ", X_lab.max(axis=0))
    print()
    print("  RGB(min): ", X_rgb.min(axis=0))
    print("  HSV(min): ", X_hsv.min(axis=0))
    print("  LAB(min): ", X_lab.min(axis=0))
    return X_hsv, X_lab, X_rgb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handle Cyclical Features

    Here, we will use `HHSV` as a term for `(Hue_sin, Hue_cos, Saturation, Value)`.
    """)
    return


@app.cell
def _(X_hsv):
    def hsv_to_hhsv(hsv):
        """
        Convert an HSV array of shape (n, 3) into an HHSV array of shape (n, 4),
        where Hue is replaced by its cyclical encoding: sin(H), cos(H).
        """
        H = hsv[:, 0]
        S = hsv[:, 1]
        V = hsv[:, 2]

        # OpenCV hue is cyclical over 180 values
        H_sin = np.sin(2 * np.pi * H / 180.0)
        H_cos = np.cos(2 * np.pi * H / 180.0)

        X_hhsv = np.column_stack((H_sin, H_cos, S, V))
        return X_hhsv

    def hhsv_to_hsv(hhsv):
        """
        Convert an HHSV array of shape (n, 4) back into an HSV array of shape (n, 3),
        where HHSV columns are [H_sin, H_cos, S, V].
        """
        H_sin = hhsv[:, 0]
        H_cos = hhsv[:, 1]
        S = hhsv[:, 2]
        V = hhsv[:, 3]

        # Recover angle in radians, range [-pi, pi]
        theta = np.arctan2(H_sin, H_cos)

        # Map to [0, 2*pi)
        theta = np.mod(theta, 2 * np.pi)

        # Convert back to OpenCV hue range [0, 180)
        H = theta * 180.0 / (2 * np.pi)

        return np.column_stack((H, S, V))

    X_hhsv = hsv_to_hhsv(X_hsv)
    return X_hhsv, hhsv_to_hsv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sanity Check

    Make sure you are able to convert all images back to their original 200x200 RGB form and display them.

    All these three images should be visually the same.
    """)
    return


@app.cell
def _(X_hhsv, X_lab, X_rgb, hhsv_to_hsv, mo):
    def hhsv_to_image(hhsv):
        hsv = hhsv_to_hsv(hhsv)
        hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        hsv = hsv.reshape(200, 200, 3)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def rgb_to_image(rgb):
        rgb = rgb.reshape(200, 200, 3)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def lab_to_image(lab):
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        lab = lab.reshape(200, 200, 3)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


    mo.hstack(
        [
            mo.image(hhsv_to_image(X_hhsv)),
            mo.image(rgb_to_image(X_rgb)),
            mo.image(lab_to_image(X_lab))
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Continue from here

    Hey you, the student! You are expected to continue from here. Train 3 k-Means models with HSV, RGB and LAB modes.
    """)
    return


if __name__ == "__main__":
    app.run()
