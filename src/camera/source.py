"""Frame ingestion: video files (primary test harness) and RTSP."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray
    frame_id: int
    timestamp: float
    camera_id: str
    source_name: str


class FrameSource:
    camera_id: str
    source_name: str
    fps: float

    def frames(self) -> Iterator[Frame]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path, camera_id: str, fps: Optional[float] = None) -> None:
        self.path = Path(path)
        self.camera_id = camera_id
        self.source_name = self.path.name
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        file_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps or file_fps or 25.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def frames(self) -> Iterator[Frame]:
        idx = 0
        while True:
            ok, image = self.cap.read()
            if not ok:
                break
            yield Frame(
                image=image,
                frame_id=idx,
                timestamp=idx / self.fps,
                camera_id=self.camera_id,
                source_name=self.source_name,
            )
            idx += 1

    def read_next(self, frame_id: int) -> Optional[Frame]:
        """Sequential read. Do not seek — CAP_PROP_POS_FRAMES is unreliable."""
        ok, image = self.cap.read()
        if not ok:
            return None
        return Frame(
            image=image,
            frame_id=frame_id,
            timestamp=frame_id / self.fps,
            camera_id=self.camera_id,
            source_name=self.source_name,
        )

    def read_at(self, frame_id: int) -> Optional[Frame]:
        """Random access. Prefer read_next() for sequential processing."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        return self.read_next(frame_id)

    def close(self) -> None:
        self.cap.release()


class RTSPSource(FrameSource):
    def __init__(
        self,
        uri: str,
        camera_id: str,
        reconnect_delay: float = 2.0,
        drop_on_overload: bool = True,
        target_fps: Optional[float] = None,
    ) -> None:
        self.uri = uri
        self.camera_id = camera_id
        self.source_name = uri
        self.reconnect_delay = reconnect_delay
        self.drop_on_overload = drop_on_overload
        self.fps = target_fps or 25.0
        self.cap = None
        self._open()

    def _open(self) -> None:
        self.cap = cv2.VideoCapture(self.uri)
        if not self.cap.isOpened():
            logger.warning("RTSP open failed for %s", self.camera_id)

    def frames(self) -> Iterator[Frame]:
        idx = 0
        last = 0.0
        min_dt = 1.0 / self.fps if self.fps else 0.0
        while True:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(self.reconnect_delay)
                self._open()
                continue
            ok, image = self.cap.read()
            if not ok:
                logger.warning("RTSP read failed camera=%s; reconnecting", self.camera_id)
                self.close()
                time.sleep(self.reconnect_delay)
                self._open()
                continue
            now = time.time()
            if self.drop_on_overload and min_dt and now - last < min_dt:
                continue
            last = now
            yield Frame(
                image=image,
                frame_id=idx,
                timestamp=now,
                camera_id=self.camera_id,
                source_name=self.source_name,
            )
            idx += 1

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_source(
    source_type: str,
    source_uri: str,
    camera_id: str,
    fps: Optional[float] = None,
) -> FrameSource:
    if source_type == "rtsp":
        return RTSPSource(source_uri, camera_id, target_fps=fps)
    return VideoFileSource(source_uri, camera_id, fps=fps)
