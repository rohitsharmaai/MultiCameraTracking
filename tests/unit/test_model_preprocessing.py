from __future__ import annotations

import numpy as np

from src.pose.rtmpose import RTMPoseBody2d
from src.segmentation.pphumanseg import PPHumanSeg


class _Input:
    name = "input"


class _Session:
    def __init__(self, outputs):
        self.outputs = outputs
        self.received = None

    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, feed):
        self.received = feed["input"]
        return self.outputs


def test_pphumanseg_uses_rgb_minus_one_to_one_normalization():
    session = _Session([np.zeros((1, 2, 192, 192), dtype=np.float32)])
    model = PPHumanSeg(session=session)
    crop = np.zeros((192, 192, 3), dtype=np.uint8)
    crop[:, :, 0] = 10  # B
    crop[:, :, 1] = 20  # G
    crop[:, :, 2] = 30  # R

    model.run(crop)

    assert session.received.shape == (1, 3, 192, 192)
    np.testing.assert_allclose(
        session.received[0, :, 0, 0],
        [(30 / 255.0 - 0.5) / 0.5, (20 / 255.0 - 0.5) / 0.5, (10 / 255.0 - 0.5) / 0.5],
        rtol=1e-6,
    )


def test_rtmpose_passes_raw_rgb_because_graph_normalizes():
    session = _Session(
        [
            np.zeros((1, 133, 384), dtype=np.float32),
            np.zeros((1, 133, 512), dtype=np.float32),
        ]
    )
    model = RTMPoseBody2d(session=session)
    crop = np.zeros((256, 192, 3), dtype=np.uint8)
    crop[:, :, 0] = 10  # B
    crop[:, :, 1] = 20  # G
    crop[:, :, 2] = 30  # R

    model.run(crop)

    assert session.received.shape == (1, 3, 256, 192)
    np.testing.assert_allclose(
        session.received[0, :, 0, 0],
        [30.0, 20.0, 10.0],
        rtol=1e-6,
    )
