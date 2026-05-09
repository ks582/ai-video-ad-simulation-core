"""T-102: Video normalization module (FR-02).

Performs audio extraction, frame extraction, and timestamp alignment.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.ad_pipeline.models import NormalizedVideo


class VideoNormalizerError(Exception):
    """Video normalization error."""


class VideoNormalizer:
    """Video normalization service.

    Acceptance criteria:
    - Can extract audio
    - Can extract frames
    - Maintains timestamp alignment
    """

    def __init__(self, output_dir: str | Path, target_fps: float = 1.0) -> None:
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps

    def normalize(self, video_path: Path) -> NormalizedVideo:
        """Normalize a video and extract audio and frame sequences."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        audio_path = self.output_dir / "audio.wav"
        probe = self._probe(video_path)

        # Extract audio
        self._extract_audio(video_path, audio_path)

        # Extract frames (at target_fps)
        total_frames = self._extract_frames(video_path, frames_dir)

        duration = float(probe.get("duration", 0))
        fps = float(probe.get("fps", self.target_fps))

        return NormalizedVideo(
            video_path=video_path,
            audio_path=audio_path,
            frames_dir=frames_dir,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration,
        )

    def _probe(self, video_path: Path) -> dict:
        """Retrieve video information using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise VideoNormalizerError(f"ffprobe execution failed: {e}") from e

        data = json.loads(result.stdout)
        info: dict = {}

        # Get duration from format
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration", 0))

        # Get fps from video stream
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                r_frame_rate = stream.get("r_frame_rate", "30/1")
                num, den = r_frame_rate.split("/")
                info["fps"] = float(num) / float(den) if float(den) != 0 else 30.0
                info["width"] = stream.get("width", 0)
                info["height"] = stream.get("height", 0)
                break

        return info

    def _extract_audio(self, video_path: Path, audio_path: Path) -> None:
        """Extract audio as WAV."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise VideoNormalizerError(f"Audio extraction failed: {e}") from e

    def _extract_frames(self, video_path: Path, frames_dir: Path) -> int:
        """Extract frames at the specified FPS."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={self.target_fps}",
            str(frames_dir / "frame_%05d.png"),
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise VideoNormalizerError(f"Frame extraction failed: {e}") from e

        return len(list(frames_dir.glob("frame_*.png")))
