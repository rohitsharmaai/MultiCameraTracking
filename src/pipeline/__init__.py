__all__ = ["FeatureExtractor", "VideoProcessor"]


def __getattr__(name: str):
    if name == "FeatureExtractor":
        from src.pipeline.features import FeatureExtractor

        return FeatureExtractor
    if name == "VideoProcessor":
        from src.pipeline.video import VideoProcessor

        return VideoProcessor
    raise AttributeError(name)
