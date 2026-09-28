import pytest

from src.observation.crop import extract_crop, xyxy_to_int
from src.types import BBox


def test_area_and_aspect():
    box = BBox(10, 20, 50, 100)
    assert box.area() == 40 * 80
    assert box.aspect_ratio() == pytest.approx(0.5)


def test_iou_identical_is_one():
    a = BBox(0, 0, 10, 10)
    assert a.iou(a) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = BBox(0, 0, 10, 10)
    b = BBox(20, 20, 30, 30)
    assert a.iou(b) == 0.0


def test_clip_and_xyxy_to_int():
    box = BBox(-5, -5, 50, 50)
    x1, y1, x2, y2 = xyxy_to_int(box, 40, 40)
    assert x1 == 0 and y1 == 0
    assert x2 == 40 and y2 == 40


def test_extract_crop_shape():
    import numpy as np

    frame = np.zeros((100, 80, 3), dtype=np.uint8)
    crop, used = extract_crop(frame, BBox(10, 10, 30, 50), pad_ratio=0.0)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert used.x1 >= 0
