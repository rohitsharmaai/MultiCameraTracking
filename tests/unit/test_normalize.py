import numpy as np

from src.reid.normalize import cosine_similarity, cosine_to_unit, l2_normalize


def test_l2_normalize_unit_length():
    v = l2_normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == np.float32(1.0) or abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_cosine_identical_is_one():
    v = [0.1, 0.2, 0.3]
    assert cosine_similarity(v, v) == np.float64(1.0) or abs(cosine_similarity(v, v) - 1.0) < 1e-5


def test_cosine_orthogonal():
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-5


def test_cosine_to_unit_range():
    assert cosine_to_unit(-1.0) == 0.0
    assert cosine_to_unit(1.0) == 1.0
    assert cosine_to_unit(0.0) == 0.5
