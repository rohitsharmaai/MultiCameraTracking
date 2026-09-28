"""Clothing-colour part descriptor fused with the OSNet body vector."""

from __future__ import annotations

import numpy as np

from src.reid.colour import FUSED_DIM, OSNET_DIM, appearance_similarity, fuse, part_colour
from tests.conftest import near, unit_vec


def _person(top_bgr, bottom_bgr, h=160, w=60):
    img = np.full((h, w, 3), 200, np.uint8)
    img[: h // 2] = top_bgr
    img[h // 2 :] = bottom_bgr
    mask = np.zeros((h, w), np.uint8)
    mask[:, 10:50] = 1
    return img, mask


def test_same_osnet_but_different_trousers_is_not_similar():
    body = unit_vec(1)
    beige = _person((20, 20, 20), (140, 180, 200))
    jeans = _person((20, 20, 20), (140, 60, 30))
    a = fuse(body, part_colour(*beige))
    b = fuse(near(body, seed=2), part_colour(*jeans))
    c = fuse(near(body, seed=3), part_colour(*beige))
    assert len(a) == FUSED_DIM
    assert appearance_similarity(a, c) > 0.8
    assert appearance_similarity(a, b) < 0.3


def test_vectors_of_different_models_are_not_compared():
    body = unit_vec(4)
    fused = fuse(body, part_colour(*_person((20, 20, 20), (140, 180, 200))))
    assert appearance_similarity(fused, body) == 0.0
    assert len(body) == OSNET_DIM
    youtu = unit_vec(5, dim=768)
    youtu_fused = fuse(youtu, part_colour(*_person((20, 20, 20), (140, 180, 200))))
    assert appearance_similarity(youtu_fused, fused) == 0.0
    assert abs(appearance_similarity(youtu_fused, youtu_fused) - 1.0) < 1e-5


def test_missing_part_is_ignored():
    body = unit_vec(5)
    img, mask = _person((20, 20, 20), (140, 180, 200))
    no_legs = mask.copy()
    no_legs[80:] = 0
    a = fuse(body, part_colour(img, mask))
    b = fuse(body, part_colour(img, no_legs))
    assert appearance_similarity(a, b) > 0.8


def test_face_vectors_are_plain_cosine():
    a, b = unit_vec(6), unit_vec(7)
    assert abs(appearance_similarity(a, b) - float(np.dot(a, b))) < 1e-5
