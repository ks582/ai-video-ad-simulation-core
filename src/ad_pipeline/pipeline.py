"""Ad preprocessing pipeline integration (FR-01 to FR-07).

Executes the entire flow from video input to Canonical Ad Schema generation.
Each ML step supports graceful degradation; processing continues with
empty results even if dependency packages are not installed.
"""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.ad_pipeline.audio_transcriber import AudioTranscriber
from src.ad_pipeline.models import (
    ASRSegment,
    AdMetadata,
    CanonicalAdSchema,
    FrameVisualTags,
    OCRResult,
    Shot,
)
from src.ad_pipeline.ocr_extractor import OCRExtractor
from src.ad_pipeline.schema_builder import CanonicalAdSchemaBuilder
from src.ad_pipeline.shot_segmenter import ShotSegmenter
from src.ad_pipeline.video_normalizer import VideoNormalizer
from src.ad_pipeline.visual_tagger import VisualTagger
from src.config_loader import get_config
from src.ingestion.ad_input_service import AdInputService

logger = logging.getLogger(__name__)


class AdPipeline:
    """Ad preprocessing pipeline.

    Input: Video file + metadata
    Output: (CanonicalAdSchema, pipeline_status)
    """

    def __init__(
        self,
        output_dir: str | Path = "data/output",
        time_bin_sec: float = 1.0,
        whisper_model: str = "base",
        ocr_languages: list[str] | None = None,
        target_fps: float = 1.0,
        transcriber: AudioTranscriber | None = None,
        ocr: OCRExtractor | None = None,
        tagger: VisualTagger | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.input_service = AdInputService()
        self.normalizer = VideoNormalizer(output_dir=self.output_dir, target_fps=target_fps)
        self.segmenter = ShotSegmenter()
        # Pass min_confidence only when the config explicitly sets it; the
        # extractor constructors carry the canonical default. This avoids
        # duplicating the threshold literal across construction paths.
        pipeline_cfg = get_config().get("pipeline", {})
        if transcriber is None:
            asr_kwargs: dict = {"model_name": whisper_model}
            if "asr_min_confidence" in pipeline_cfg:
                asr_kwargs["min_confidence"] = float(pipeline_cfg["asr_min_confidence"])
            transcriber = AudioTranscriber(**asr_kwargs)
        self.transcriber = transcriber
        if ocr is None:
            ocr_kwargs: dict = {"languages": ocr_languages}
            if "ocr_min_confidence" in pipeline_cfg:
                ocr_kwargs["min_confidence"] = float(pipeline_cfg["ocr_min_confidence"])
            ocr = OCRExtractor(**ocr_kwargs)
        self.ocr = ocr
        self.tagger = tagger or VisualTagger()
        self.schema_builder = CanonicalAdSchemaBuilder(time_bin_sec=time_bin_sec)

    def run(
        self,
        video_path: str | Path,
        metadata: AdMetadata,
        progress_callback=None,
    ) -> tuple[CanonicalAdSchema, dict[str, str]]:
        """Run the pipeline.

        Returns:
            A tuple of (schema, pipeline_status).
            pipeline_status is a dict recording the result of each step.
            Values are "ok" (success) or "skipped: <reason>" (skipped).
        """
        status: dict[str, str] = {}
        _t_pipeline = _time.time()

        def _progress(step, label, percent):
            if progress_callback:
                progress_callback(step, label, percent)

        # 1. Ad input registration (required)
        _progress("extracting_frames", "Extracting frames...", 5)
        logger.info("Phase 1/7: input registration — started")
        _t = _time.time()
        ad_input = self.input_service.register(video_path, metadata)
        status["input"] = "ok"
        logger.info("Phase 1/7: input registration — done (%.1fs)", _time.time() - _t)

        # 2. Video normalization (required -- ffmpeg/ffprobe assumed available)
        _progress("normalizing_video", "Normalizing video...", 10)
        logger.info("Phase 2/7: video normalization — started")
        _t = _time.time()
        normalized = self.normalizer.normalize(ad_input.video_path)
        status["normalize"] = "ok"
        logger.info("Phase 2/7: video normalization — done (%.1fs)", _time.time() - _t)

        # 3. Shot segmentation (optional). Boundary detection only —
        # PR 2 of the Option 3 refactor demoted ``Shot`` to a temporal
        # annotation; downstream ML stages (OCR / visual tagging) now
        # consume the full 1fps frame set, so we no longer need to
        # populate ``representative_frame_paths`` to gate inference.
        _progress("shot_segmentation", "Detecting shots...", 15)
        logger.info("Phase 3/7: shot segmentation — started")
        _t = _time.time()
        try:
            shots = self.segmenter.segment(ad_input.video_path)
            status["shot_segmentation"] = "ok"
            logger.info("Phase 3/7: shot segmentation — done (%.1fs, %d shots)", _time.time() - _t, len(shots))
        except Exception as e:
            logger.warning("Phase 3/7: shot segmentation — skipped (%.1fs): %s", _time.time() - _t, e)
            shots = [Shot(shot_id=0, start_sec=0.0, end_sec=normalized.duration_sec)]
            status["shot_segmentation"] = f"skipped: {e}"

        # Collect all 1fps-extracted frames; the new ML stages tag every
        # frame rather than a single representative per shot. Sorted by
        # filename so the resulting per-frame lists align with bin order
        # (frame_NNNNN.png is zero-padded so lexical sort == numeric sort
        # for any video shorter than ~28 hours, well above the 60s cap).
        all_frames = sorted(normalized.frames_dir.glob("frame_*.png"))

        # 4-6. ASR, OCR, Visual tagging — independent steps, run in parallel
        _progress("analyzing", "Analyzing audio, text, and visuals...", 25)
        asr_segments: list[ASRSegment] = []
        ocr_results: list[OCRResult] = []
        frame_visual_tags: list[FrameVisualTags] = []

        # Tier A — parallel ML-model loading. The earlier strictly-sequential
        # path was a defensive guard against Python's per-module
        # ``_ModuleLock`` contention on first imports; we keep that guarantee
        # by forcing the three top-level modules to load on the main thread
        # FIRST (cheap — module registration only, no weight I/O) and then
        # running the heavy ``disk → RAM`` work concurrently across whisper /
        # easyocr / open_clip. Each underlying loader releases the GIL
        # during file I/O, so three workers yield roughly ``max(t_whisper,
        # t_easyocr, t_clip)`` instead of their sum — observed on Fly.io
        # shared-cpu-1x: 168 s → ~80 s.
        logger.info("Phase 4-6: loading ML models — started")
        _t = _time.time()

        # Step 1: warm-import the three top-level modules on the main
        # thread. Wrapped in ``try/except ImportError`` so an installation
        # missing one of the optional deps still degrades gracefully — the
        # subsequent ``_load_model`` call will raise the canonical
        # ``*Error`` and the pipeline's per-stage status logging picks it
        # up the same way the sequential path used to.
        for _mod_name in ("whisper", "easyocr", "open_clip"):
            try:
                __import__(_mod_name)
            except ImportError:
                # Defer the user-facing failure to the loader itself; it
                # raises a domain-specific error with installation guidance.
                pass

        # Step 2: run the actual model construction concurrently. Any
        # exception from a worker propagates via ``Future.result()`` and
        # is handled by the surrounding try/except wrappers below
        # (asr / ocr / visual_tagging status fields).
        with ThreadPoolExecutor(max_workers=3) as _load_pool:
            _load_futs = [
                _load_pool.submit(self.transcriber._load_model),
                _load_pool.submit(self.ocr._load_reader),
                _load_pool.submit(self.tagger._load_model),
            ]
            for _f in _load_futs:
                _f.result()

        logger.info("Phase 4-6: loading ML models — done (%.1fs)", _time.time() - _t)

        logger.info("Phase 4-6: ASR + OCR + Visual tagging (parallel) — started")
        _t_parallel = _time.time()

        # ``frame_extraction_fps`` is the rate at which frames were
        # written to ``frames_dir`` (controlled by ``VideoNormalizer.target_fps``,
        # default 1.0). ``normalized.fps`` reports the *source* video rate
        # (e.g. 24 fps), which would map frame_idx 2 to t=0.042s instead
        # of t=1.0s. Use the extraction rate so ``time_sec`` aligns with
        # the 1-second ``time_bins`` granularity.
        frame_extraction_fps = self.normalizer.target_fps
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_asr = executor.submit(self.transcriber.transcribe, normalized.audio_path)
            fut_ocr = executor.submit(
                self.ocr.extract_from_frames, all_frames, frame_extraction_fps
            )
            fut_tags = executor.submit(
                self.tagger.tag_frames,
                all_frames,
                brand=ad_input.metadata.brand,
                category=ad_input.metadata.category,
                fps=frame_extraction_fps,
            )

            try:
                asr_segments = fut_asr.result()
                status["asr"] = "ok"
                logger.info("Phase 4/7: ASR — done (%d segments)", len(asr_segments))
            except Exception as e:
                logger.warning("Phase 4/7: ASR — skipped: %s", e)
                status["asr"] = f"skipped: {e}"

            try:
                ocr_results = fut_ocr.result()
                status["ocr"] = "ok"
                logger.info("Phase 5/7: OCR — done (%d results)", len(ocr_results))
            except Exception as e:
                logger.warning("Phase 5/7: OCR — skipped: %s", e)
                status["ocr"] = f"skipped: {e}"

            try:
                frame_visual_tags = fut_tags.result()
                status["visual_tagging"] = "ok"
                logger.info("Phase 6/7: Visual tagging — done (%d frame tags)", len(frame_visual_tags))
            except Exception as e:
                logger.warning("Phase 6/7: Visual tagging — skipped: %s", e)
                # Non-empty fallback: emit one no-content FrameVisualTags
                # per extracted frame so the time-bin builder sees the
                # expected list shape and downstream metrics treat the
                # frames as no-content rather than missing.
                frame_visual_tags = [
                    FrameVisualTags(
                        frame_idx=idx + 1,
                        time_sec=idx / max(frame_extraction_fps, 1.0),
                    )
                    for idx in range(len(all_frames))
                ]
                status["visual_tagging"] = f"skipped: {e}"

        logger.info("Phase 4-6: parallel analysis — done (%.1fs)", _time.time() - _t_parallel)
        _progress("analyzing_complete", "Analysis complete...", 45)

        # 7. Canonical Ad Schema generation (required)
        _progress("building_schema", "Building ad schema...", 50)
        logger.info("Phase 7/7: schema generation — started")
        _t = _time.time()
        schema = self.schema_builder.build(
            input_id=ad_input.input_id,
            metadata=ad_input.metadata,
            shots=shots,
            asr_segments=asr_segments,
            ocr_results=ocr_results,
            frame_visual_tags=frame_visual_tags,
        )
        status["schema"] = "ok"
        logger.info("Phase 7/7: schema generation — done (%.1fs)", _time.time() - _t)

        logger.info("Pipeline complete (%.1fs total)", _time.time() - _t_pipeline)
        return schema, status
