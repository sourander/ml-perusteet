import random

from dataclasses import dataclass
from ml.utils.math import euclidean_distance

@dataclass
class Point:
    coordinates: tuple[float]
    cluster_id: int = -1

@dataclass
class Centroid:
    coordinates: tuple[float]
    id: int

class kMeans:
    def __init__(self, k: int, seed: int | None = None):
        # Hyperparameters
        self.k = k
        self._rng = random.Random(seed)

        # State
        self.iter = 0
        self._is_converged = False
        self._converted_reason = ""
        self.k_is_stopped = {k: False for k in range(k)}

        # Data
        self.centroids: list[Centroid] = []
        self.points: list[Point] = []

    def initialize_centroids(self) -> list[Centroid]:
        # Pick k random points as initial centroids
        k_points = self._rng.sample(self.points, self.k)
        return [Centroid(coordinates=point.coordinates, id=i) for i, point in enumerate(k_points)]

    @staticmethod
    def data_to_points(data: list[tuple[float]]) -> list[Point]:
        return [Point(coordinates=tuple(point)) for point in data]
    
    @staticmethod
    def mean_coordinates(coordinates: list[list[float]]) -> list[float]:
        """Compute the mean location of a list of coordinates.

        Parameters:
        coordinates: 
          A list of coordinates. (Contents of Point.coordinates)
        
        Usage:
        >>> mean_location([[1, 2], [2, 3], [3, 4]])
        [2.0, 3.0]
        """
        return [sum(x) / len(coordinates) for x in zip(*coordinates)]
    
    def assign_cluster(self, point: Point):
        distances = [euclidean_distance(point.coordinates, centroid.coordinates) for centroid in self.centroids]
        point.cluster_id = distances.index(min(distances))

    def update_centroid(self, centroid: Centroid):
        cluster_points = [point.coordinates for point in self.points if point.cluster_id == centroid.id]

        if not cluster_points:
            self.k_is_stopped[centroid.id] = True
        
        if cluster_points:
            new_features = self.mean_coordinates(cluster_points)
            
            # Vote for stopping
            if centroid.coordinates == new_features:
                self.k_is_stopped[centroid.id] = True
            else:
                self.k_is_stopped[centroid.id] = False
                centroid.coordinates = new_features

    def fit(self, data: list[tuple[float]], max_iter: int = 100):

        if len(data) <= self.k:
            raise ValueError("Number of clusters k should be less than the number of data points")
        
        # Make Points
        self.points = self.data_to_points(data)

        # Initialize Centroids
        self.centroids = self.initialize_centroids()

        # Counter
        iter = 0

        # Main Loop
        while True:
            for point in self.points:
                self.assign_cluster(point)

            for centroid in self.centroids:
                self.update_centroid(centroid)

            iter += 1
            if iter >= max_iter:
                self._is_converged = True
                self._converted_reason = f"Reached max_iter of {max_iter}"

            # Check if all centroids have converged
            if all(self.k_is_stopped.values()):
                self._is_converged = True
                self._converted_reason = "All centroids have converged"

            if self._is_converged:
                break

    def predict(self, data: list[tuple[float]]) -> list[int]:

        if not self._is_converged:
            raise ValueError("Model has not been fitted yet")
        
        if len(data[0]) != len(self.centroids[0].coordinates):
            raise ValueError("Data has different number of features than the model")

        points = self.data_to_points(data)
        for point in points:
            self.assign_cluster(point)
        return [point.cluster_id for point in points]
    
    def summary(self):

        def round_list(lst):
            return [round(x, 2) for x in lst]

        coordinates = {centroid.id: round_list(centroid.coordinates) for centroid in self.centroids}

        return {
            **coordinates,
            "iterations": self.iter,
            "converged": self._is_converged,
            "reason": self._converted_reason
        }


if __name__ == "__main__":

    import random
    import json

    random.seed(42)

    # Note that the ml.utils.math.euclidean_distance function is available only in the test environment
    # since there the ml package is actually vastaukset/ package. To be able to run this locally,
    # we will simply overwrite the function here.
    def euclidean_distance(p, q) -> float:
        return sum((p - q) ** 2 for p, q in zip(p, q)) ** 0.5

    def f(center, noise=0.5):
        return center + random.gauss(-noise, noise)

    # Randomize data around (3,3,3) and (-3,-3,-3) and (-4,-4,5) locations
    data_k1 = [(f(1), f(1), f(1)) for _ in range(100)]
    data_k2 = [(f(-1), f(-1), f(-1)) for _ in range(100)]
    data_k3 = [(f(-1), f(-1), f(1)) for _ in range(100)]
    data = data_k1 + data_k2 + data_k3

    model = kMeans(k=3, seed=42)
    model.fit(data)

    # There is a stats method to get the stats for helping debugging
    print(json.dumps(model.summary(), indent=2))

    reference_locations = [
        (1, 1, 1),
        (-1, -1, -1),
        (-1, -1, 1),
    ]

    # Check that at the centroids are around the original locations
    for centroid in model.centroids:
        for ref in reference_locations:

            # Check that the distance is less than 0.5
            # and then pop out the reference location
            dist = euclidean_distance(ref, centroid.coordinates)
            if dist < 1.5:
                reference_locations.remove(ref)
                print(f"Centroid {centroid.id} is around {ref} with distance {dist}")
                continue


    # If all reference locations are popped out, then the test is passed
    assert not reference_locations, f"Reference location {reference_locations} did not find a match"

    # You can predict any number of data points
    test_data = [[1, 1, 1], [-1, -1, -1], [-1, 1, 1]]
    print(model.predict(test_data))