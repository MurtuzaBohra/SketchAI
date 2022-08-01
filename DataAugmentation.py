import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter
from HelperFunctions import convert_curve_points_to_svg


def random_rotate_sample(sample, class_name):
    # We'll do 2 possible rotations:
    ## small rotation around zero
    ## large 2.pi rotation
    ## The rationale is that sometimes 2.pi rotations are not good for some shapes (e.g rectangle)

    # small_rotation = random.uniform(0, 1) < 0.2
    small_rotation = True
    if class_name in ["circle", 'dot', 'arrow', 'arrow_head', 'triangle']:
        small_rotation = False

    if small_rotation:
        theta = random.uniform(-math.pi / 40, math.pi / 40)
    else:
        theta = random.uniform(0, 2.0 * math.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.transpose(R @ np.transpose(sample))


def random_symmetry_sample(sample, className):
    if className in ['curly_braces_e', 'bracket_w', 'bracket_n', 'curly_braces_w', 'bracket_s', 'curly_braces_n', 'curly_braces_s', 'bracket_e']:
        return sample
    transformation = random.randrange(0, 4)
    if transformation == 0:
        return sample * [-1.0, -1.0]
    if transformation == 1:
        return sample * [1.0, -1.0]
    if transformation == 2:
        return sample * [1.0, -1.0]
    return sample


# Elastic transform
#
# This is a costly transform if we compute a transform map for every stroke
# We can pre-compute some maps, and then select on randomly


class ElasticTransform:
    def __init__(
            self, alpha=0.1, sigma=5.0, shape=(100, 100), nb_maps=25, random_state=None
    ):
        self.maps = [
            self.compute_map(alpha, sigma, shape, random_state) for _ in range(nb_maps)
        ]

    def apply(self, points):
        map_index = random.randint(0, len(self.maps) - 1)
        return self.transform(points, *self.maps[map_index])

    def compute_map(self, alpha, sigma, shape, random_state):
        if random_state is None:
            random_state = np.random.RandomState(None)
        dx = (
                gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
                )
                * alpha
        )
        dy = (
                gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
                )
                * alpha
        )
        return (dx, dy)

    def transform(self, points, dx, dy):
        x_min, y_min = np.nanmin(points, axis=0, keepdims=True)[0]
        x_max, y_max = np.nanmax(points, axis=0, keepdims=True)[0]
        max_length = max(*dx.shape)
        coordinates = np.clip(
            np.floor(
                max_length
                / max(x_max - x_min, y_max - y_min)
                * (points - [x_min, y_min])
            ),
            0,
            max_length - 1,
        ).astype(int)
        return points + np.array([[dx[y][x], dy[y][x]] for x, y in coordinates])


def translate_to_topLeft(points):
    min_xy = np.min(points, axis=0)
    return np.array(points, dtype=float) - min_xy


class DataAugmentation:
    def __init__(self, augmentFactor=2):
        self.elasticTransform = ElasticTransform()
        self.augmentFactor = augmentFactor

    def apply_data_transform(self, sample, className=''):  # sample-> [#points on stroke, 2]
        samples = []
        for i in range(self.augmentFactor):
            if random.uniform(0, 1) < 0.5:
                sample = self.elasticTransform.apply(sample)
            if random.uniform(0, 1) < 0.5:
                sample = random_rotate_sample(sample, className)
            if random.uniform(0, 1) < 0.5:
                sample = random_symmetry_sample(sample, className)
            sample = translate_to_topLeft(sample)
            samples.append(sample)
        return samples


if __name__ == "__main__":
    sample = [34, 242, 5380, 34, 238, 5396.3623046875, 33, 180.75, 5630.5869140625, 32.5, 152.125, 5747.69921875, 32,
              123.5, 5864.8115234375, 32, 35, 6226.8330078125, 30, 13, 6317.1982421875, 30, 9, 6333.5615234375, 30, 5,
              6349.923828125, 28, 2, 6364.6728515625, 22, 2.25, 6389.2373046875, 19, 2.375, 6401.5205078125, 16, 2.5,
              6413.802734375, 9, 1.25, 6442.890625, 5.5, 0.625, 6457.43359375, 3.75, 0.3125, 6464.7060546875, 2, 0,
              6471.9775390625, 2, 2, 6480.1591796875, 0, 32, 6603.150390625, 0, 111, 6926.310546875, 2.5, 121.5,
              6970.462890625, 3, 229, 7410.2119140625, 3, 236, 7438.845703125, 3, 239.5, 7453.1630859375, 3, 243,
              7467.48046875, 6, 243, 7479.751953125, 14, 240, 7514.703125, 17, 240.5, 7527.1435546875, 20, 241,
              7539.5849609375, 27, 241.5, 7568.29296875, 30.5, 241.75, 7582.646484375, 32.25, 241.875, 7589.8232421875,
              34, 242, 7597]
    sample = np.array(sample).reshape((-1, 3))
    sample = sample[:, :2]
    dataAugmentation = DataAugmentation()
    samples = dataAugmentation.apply_data_transform(sample)
    convert_curve_points_to_svg(samples[0])
