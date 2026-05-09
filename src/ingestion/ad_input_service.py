"""T-101: Ad input module (FR-01).

Handles video file registration, metadata registration, and input ID assignment.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from src.ad_pipeline.models import AdInput, AdMetadata

_SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class AdInputError(Exception):
    """Ad input error."""


class AdInputService:
    """Ad input service.

    Acceptance criteria:
    - Accepts mp4 files
    - Returns an error when metadata is missing
    - Assigns an input ID
    """

    def register(
        self,
        video_path: str | Path,
        metadata: AdMetadata,
    ) -> AdInput:
        """Register an ad video and metadata, and assign an input ID."""
        video_path = Path(video_path)

        # Check file existence
        if not video_path.exists():
            raise AdInputError(f"Video file not found: {video_path}")

        # Format check
        if video_path.suffix.lower() not in _SUPPORTED_FORMATS:
            raise AdInputError(
                f"Unsupported format: {video_path.suffix}. "
                f"Supported: {_SUPPORTED_FORMATS}"
            )

        # Assign input ID
        input_id = f"ad-{uuid.uuid4().hex[:12]}"

        return AdInput(
            input_id=input_id,
            video_path=video_path,
            metadata=metadata,
        )
