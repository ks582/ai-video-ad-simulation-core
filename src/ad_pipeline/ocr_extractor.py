"""T-105: OCR module (FR-05).

Extracts on-screen text from every 1 fps frame produced by the
``VideoNormalizer``. The previous shot-level entry point
(``extract_from_shots``) was removed in PR 3 of the Option 3 refactor
(2026-05-09); see ``ai_ad_simulation_requirements2_updated.md`` §1.16.x.
"""

from __future__ import annotations

from pathlib import Path

from src.ad_pipeline.models import OCRResult

# Hard floor below which OCR output is treated as garbage even if the caller
# requests a lower threshold. EasyOCR's score distribution is heavy-tailed at
# the bottom — anything in the 1e-3 range is essentially random noise.
_ABSOLUTE_MIN_CONFIDENCE = 0.05


class OCRExtractorError(Exception):
    """OCR error."""


class OCRExtractor:
    """OCR extraction service.

    Acceptance criteria:
    - Retains text strings and their timestamps
    - Saves in a format that allows deduplication of OCR results downstream
    - Drops detections with confidence below ``min_confidence`` so noisy
      misreadings (e.g. EasyOCR scoring 0.0002 on a non-text region) do not
      leak into the schema or downstream effect estimation.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        min_confidence: float = 0.30,
    ) -> None:
        """
        Args:
            languages: List of OCR target languages (e.g. ["ja", "en"]).
            min_confidence: Drop detections whose confidence is strictly less
                than this. Defaults to 0.30 — the practical floor below which
                EasyOCR outputs are dominated by garbage. Values below
                ``_ABSOLUTE_MIN_CONFIDENCE`` are clamped up to that floor so
                callers cannot disable filtering by mistake.
        """
        self.languages = languages or ["en"]
        self.min_confidence = max(float(min_confidence), _ABSOLUTE_MIN_CONFIDENCE)
        self._reader = None

    def _accept(self, text: str, conf: float) -> bool:
        """Filter predicate applied uniformly to every EasyOCR detection."""
        if conf < self.min_confidence:
            return False
        if not text or not text.strip():
            return False
        return True

    @staticmethod
    def _build_result(
        text: str,
        bbox,
        conf: float,
        time_sec: float,
        frame_path: Path,
    ) -> OCRResult:
        return OCRResult(
            text=text.strip(),
            bbox=[[int(p[0]), int(p[1])] for p in bbox],
            time_sec=time_sec,
            confidence=float(conf),
            frame_path=frame_path,
        )

    def _load_reader(self):
        """Lazy-load the EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
            except ImportError as e:
                raise OCRExtractorError(
                    "easyocr is not installed: pip install easyocr"
                ) from e
            self._reader = easyocr.Reader(self.languages, gpu=False)
        return self._reader

    def extract_from_frames(
        self,
        frame_paths: list[Path],
        fps: float,
    ) -> list[OCRResult]:
        """Run OCR on a list of frames (computing timestamps from frame numbers)."""
        reader = self._load_reader()
        results: list[OCRResult] = []

        for frame_path in frame_paths:
            if not frame_path.exists():
                continue
            # frame_00001.png -> index 0 -> time 0.0
            stem = frame_path.stem
            idx = int(stem.split("_")[-1]) - 1  # 1-indexed to 0-indexed
            time_sec = idx / fps if fps > 0 else 0.0

            for bbox, text, conf in reader.readtext(str(frame_path)):
                if not self._accept(text, float(conf)):
                    continue
                results.append(self._build_result(text, bbox, conf, time_sec, frame_path))

        return results
