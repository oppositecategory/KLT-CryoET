import numpy as np

from kltpicker_3d.utils import ranked_local_maxima_nms_3d


def test_nms_does_not_create_suppression_boundary_candidates():
    coordinates = np.indices((15, 15, 15))
    center = np.array([5, 5, 5])
    distance = np.sqrt(np.sum((coordinates - center[:, None, None, None]) ** 2, axis=0))
    scores = 100.0 - distance
    scores[12, 12, 12] = 99.0

    indices, values = ranked_local_maxima_nms_3d(
        scores,
        radius=3.0,
        max_picks=5,
    )

    np.testing.assert_array_equal(indices, np.array([[5, 5, 5], [12, 12, 12]]))
    np.testing.assert_allclose(values, np.array([100.0, 99.0]))


def test_nms_suppresses_weaker_original_local_maximum_inside_radius():
    scores = np.zeros((11, 11, 11), dtype=np.float64)
    scores[3, 3, 3] = 10.0
    scores[3, 3, 5] = 9.0
    scores[8, 8, 8] = 8.0

    indices, values = ranked_local_maxima_nms_3d(
        scores,
        radius=2.5,
        max_picks=3,
    )

    np.testing.assert_array_equal(indices, np.array([[3, 3, 3], [8, 8, 8]]))
    np.testing.assert_allclose(values, np.array([10.0, 8.0]))
