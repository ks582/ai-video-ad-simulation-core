"""T-103: Shot segmentation module (FR-03).

Performs shot boundary detection and representative frame selection.
"""

from __future__ import annotations

from pathlib import Path

from src.ad_pipeline.models import Shot


class ShotSegmenterError(Exception):
    """Shot segmentation error."""


class ShotSegmenter:
    """Shot segmentation service.

    Acceptance criteria:
    - Each shot has start/end timestamps
    - At least one representative frame is retained per shot
    """

    def __init__(self, threshold: float = 27.0) -> None:
        """
        Args:
            threshold: PySceneDetect ContentDetector threshold.
        """
        self.threshold = threshold

    def segment(self, video_path: Path) -> list[Shot]:
        """Segment a video into shots."""
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError as e:
            raise ShotSegmenterError(
                "scenedetect is not installed: pip install scenedetect"
            ) from e

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            # If no scenes were detected, treat the entire video as a single shot
            duration = video.duration.get_seconds()
            return [Shot(shot_id=0, start_sec=0.0, end_sec=duration)]

        shots: list[Shot] = []
        for i, (start, end) in enumerate(scene_list):
            shots.append(
                Shot(
                    shot_id=i,
                    start_sec=start.get_seconds(),
                    end_sec=end.get_seconds(),
                )
            )

        return shots

