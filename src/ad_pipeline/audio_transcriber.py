"""T-104: ASR / diarization module (FR-04).

Performs audio transcription and speaker diarization.
"""

from __future__ import annotations

from pathlib import Path

from src.ad_pipeline.models import ASRSegment

# Hard floor below which Whisper segments are treated as garbage even if the
# caller requests a lower threshold. Whisper's avg_logprob distribution can
# produce sub-0.05 confidence scores for music-only or pure-noise frames,
# which are not real narration.
_ABSOLUTE_MIN_CONFIDENCE = 0.05


class AudioTranscriberError(Exception):
    """Audio transcription error."""


class AudioTranscriber:
    """Audio transcription service.

    Acceptance criteria:
    - Each utterance has start/end timestamps
    - Can retain text and speaker information
    - Drops segments whose ``avg_logprob``-derived confidence is below
      ``min_confidence`` so ambient sound / music misreadings (e.g. Whisper
      transcribing water sounds as repeating "1.5%" at confidence 0.20)
      do not leak into the schema or downstream effect estimation.
    """

    def __init__(
        self,
        model_name: str = "base",
        language: str | None = None,
        min_confidence: float = 0.30,
    ) -> None:
        """
        Args:
            model_name: Whisper model size (tiny/base/small/medium/large).
            language: Language code (None for auto-detection).
            min_confidence: Drop segments whose confidence is strictly less
                than this. Defaults to 0.30 — the practical floor below which
                Whisper outputs are dominated by ambient-sound misreadings.
                Values below ``_ABSOLUTE_MIN_CONFIDENCE`` are clamped up so
                callers cannot disable filtering by mistake.
        """
        self.model_name = model_name
        self.language = language
        self.min_confidence = max(float(min_confidence), _ABSOLUTE_MIN_CONFIDENCE)
        self._model = None

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            try:
                import whisper
            except ImportError as e:
                raise AudioTranscriberError(
                    "whisper is not installed: pip install openai-whisper"
                ) from e
            self._model = whisper.load_model(self.model_name)
        return self._model

    def _accept(self, text: str, conf: float) -> bool:
        """Filter predicate applied uniformly to every Whisper segment."""
        if conf < self.min_confidence:
            return False
        if not text or not text.strip():
            return False
        return True

    def transcribe(self, audio_path: Path) -> list[ASRSegment]:
        """Transcribe an audio file."""
        if not audio_path.exists():
            raise AudioTranscriberError(f"Audio file not found: {audio_path}")

        model = self._load_model()
        options = {}
        if self.language:
            options["language"] = self.language

        result = model.transcribe(str(audio_path), **options)

        segments: list[ASRSegment] = []
        for seg in result.get("segments", []):
            # Whisper's avg_logprob is negative (-inf to 0), so clamp to 0-1
            raw_conf = float(seg.get("avg_logprob", -1.0))
            confidence = max(0.0, min(1.0, 1.0 + raw_conf))

            text = seg["text"].strip()
            if not self._accept(text, confidence):
                continue

            segments.append(
                ASRSegment(
                    start_sec=float(seg["start"]),
                    end_sec=float(seg["end"]),
                    text=text,
                    speaker_label=str(seg.get("speaker", "")),
                    confidence=confidence,
                )
            )

        return segments