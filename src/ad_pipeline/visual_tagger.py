"""T-106: Visual tag extraction module (FR-06).

Outputs logo exposure, product exposure, CTA presence, and emotion tags as structured tags.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.ad_pipeline.models import FrameVisualTags

logger = logging.getLogger(__name__)


class VisualTaggerError(Exception):
    """Visual tag extraction error."""


# Category definitions for zero-shot classification (multiple prompts for improved accuracy)
_SCENE_PROMPTS: dict[str, list[str]] = {
    "outdoor": ["an outdoor scene", "outside scenery"],
    "indoor": ["an indoor scene", "inside a room"],
    "studio": ["a studio setting", "a professional studio"],
    "nature": ["a nature scene", "natural landscape"],
    "urban": ["an urban scene", "a city street"],
    "product_closeup": ["a product closeup", "a product showcase"],
    "lifestyle": ["a lifestyle scene", "people in everyday life"],
    "text_overlay": ["text overlay on video", "advertising text on screen"],
    "animation": ["an animated scene", "computer generated graphics"],
}

_EMOTION_PROMPTS: dict[str, list[str]] = {
    "happy": ["a happy joyful scene", "people smiling"],
    "exciting": ["an exciting dynamic scene", "action and energy"],
    "calm": ["a calm peaceful scene", "serene atmosphere"],
    "serious": ["a serious scene", "formal atmosphere"],
    "humorous": ["a funny humorous scene", "comedy"],
    "dramatic": ["a dramatic intense scene", "dramatic lighting"],
    "nostalgic": ["a nostalgic vintage scene", "retro atmosphere"],
    "inspiring": ["an inspiring motivational scene", "empowering moment"],
}

_OBJECT_PROMPTS: dict[str, list[str]] = {
    "logo": ["a corporate brand logo symbol", "a brand emblem or badge"],
    "product": ["a product being showcased for sale", "an advertised product on display"],
    "cta_button": ["a buy now or sign up button overlay", "a call to action banner on advertisement"],
    "text_banner": ["text overlay on video", "advertising text on screen", "promotional text"],
    "person": ["a person", "a human face", "people"],
    "food": ["food", "a meal", "eating"],
    "vehicle": ["a car", "an automobile", "a vehicle"],
}


_MAX_BRAND_TERM_LEN = 64  # truncation guard for prompts; CLIP token budget is ~77


def _sanitize_brand_term(term: str) -> str:
    """Normalize a brand/category string for safe inclusion in CLIP prompts.

    - Removes punctuation/control chars that would confuse CLIP tokenization
    - Collapses runs of whitespace (e.g. "Home & Garden" → "home garden", not "home  garden")
    - Truncates to ``_MAX_BRAND_TERM_LEN`` characters to bound prompt size
    - Lowercases for prompt consistency

    Note: this function defends only against tokenization/length issues. It is
    NOT a defense against semantic prompt injection. Downstream model integrations
    should wrap extracted labels as untrusted data before using them in prompts.
    """
    if not term:
        return ""
    cleaned = re.sub(r"[^\w\s\-]", "", term)
    cleaned = re.sub(r"-{2,}", "-", cleaned)  # collapse "--" sequences
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > _MAX_BRAND_TERM_LEN:
        cleaned = cleaned[:_MAX_BRAND_TERM_LEN].rstrip()
    return cleaned.lower()


def build_object_prompts(brand: str = "", category: str = "") -> dict[str, list[str]]:
    """Return object prompts with optional brand/category injection.

    Adds prompts like "the {brand} logo" and "a {category} product" so that
    CLIP zero-shot can leverage brand-specific visual priors. The original
    generic prompts are retained as fallbacks.
    """
    prompts = {k: list(v) for k, v in _OBJECT_PROMPTS.items()}
    b = _sanitize_brand_term(brand)
    c = _sanitize_brand_term(category)
    if b:
        prompts["logo"].extend([
            f"the {b} logo",
            f"a {b} branded product",
        ])
        prompts["product"].extend([
            f"a {b} product",
            f"{b} packaging on display",
        ])
    if c:
        prompts["product"].append(f"a {c} product")
    return prompts


class VisualTagger:
    """Visual tag extraction service.

    Acceptance criteria:
    - Output follows a fixed schema
    - Uses structured tags rather than free-form text
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        # Batch size for ``tag_frames`` CLIP inference. Default of 8 is
        # tuned for shared-cpu-1x (2 GB RAM) — ViT-B/32 weights ~380 MB
        # plus an 8-frame batch of 224x224x3 preprocessed tensors stays
        # under the 1 GB peak budget that has to coexist with EasyOCR
        # (~300 MB) running in parallel on the same container.
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the CLIP model."""
        if self._model is None:
            try:
                import open_clip
            except ImportError as e:
                raise VisualTaggerError(
                    "open-clip-torch is not installed: pip install open-clip-torch"
                ) from e
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        return self._model, self._preprocess, self._tokenizer

    def _classify_frame(
        self,
        frame_path: Path,
        object_prompts: dict[str, list[str]] | None = None,
        scene_threshold: float = 0.24,
        emotion_threshold: float = 0.25,
        object_threshold: float = 0.27,
    ) -> dict[str, list[str]]:
        """Classify a frame using zero-shot classification (cosine similarity based)."""
        import torch
        from PIL import Image

        model, preprocess, tokenizer = self._load_model()
        prompts = object_prompts if object_prompts is not None else _OBJECT_PROMPTS

        image = preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0)

        result: dict[str, list[str]] = {}

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Scene tags
            result["scenes"] = self._zero_shot_multi(
                image_features, _SCENE_PROMPTS, tokenizer, model, scene_threshold
            )
            # Emotion tags
            result["emotions"] = self._zero_shot_multi(
                image_features, _EMOTION_PROMPTS, tokenizer, model, emotion_threshold
            )
            # Object tags (brand-aware when prompts include brand/category)
            result["objects"] = self._zero_shot_multi(
                image_features, prompts, tokenizer, model, object_threshold
            )

        return result

    @staticmethod
    def _zero_shot_multi(image_features, label_prompts, tokenizer, model, threshold):
        """Zero-shot classify with multiple prompts, returning labels whose max cosine similarity exceeds the threshold."""
        import torch

        matched = []
        for label, prompts in label_prompts.items():
            text_tokens = tokenizer(prompts)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)
            max_sim = similarity.max().item()
            if max_sim >= threshold:
                matched.append(label)

        return matched

    # ------------------------------------------------------------------
    # Option 3 — frame-level inference path
    #
    # ``tag_frames`` and its batch helpers below run CLIP on every
    # supplied frame rather than on a single representative frame per
    # shot. The shot-level ``tag_shots`` predecessor was removed in PR 3
    # of the Option 3 refactor (2026-05-09). See
    # ``ai_ad_simulation_requirements2_updated.md`` §1.16.x for the
    # design rationale (root cause: shot-detector returning 1 shot for
    # an 8 s AI-generated ad caused the visual signal to be sampled at
    # only the mid-frame, missing both the opening logo card and the
    # end-card CTA).
    # ------------------------------------------------------------------

    _FRAME_IDX_RE = re.compile(r"(\d+)")

    @classmethod
    def _parse_frame_idx(cls, frame_path: Path) -> int:
        """Extract the 1-indexed frame number from a ``frame_NNNNN.png`` filename.

        Defends against a future change where the normalizer's filename
        scheme drifts (e.g. wider zero-padding) by parsing the trailing
        digit run rather than relying on list position. Mirrors the
        approach in ``ocr_extractor.py`` so the two pipelines extract
        identical frame indices for the same file.
        """
        m = cls._FRAME_IDX_RE.search(frame_path.stem)
        if not m:
            raise VisualTaggerError(
                f"Cannot parse frame index from {frame_path.name!r}; "
                f"expected a filename containing a digit run such as "
                f"'frame_00001.png'."
            )
        return int(m.group(1))

    def _encode_label_prompts(
        self,
        label_prompts: dict[str, list[str]],
    ) -> dict[str, "object"]:
        """Encode each label's prompts once and return normalized text features.

        Risk-1 mitigation per the architect's design memo: encode every
        text prompt up-front (one pass per ``tag_frames`` call) and reuse
        the resulting tensors across all image batches. This eliminates
        any chance that successive ``tokenizer()`` invocations introduce
        per-call state that subtly changes the text features mid-run.

        Returns:
            dict mapping label -> ``(P, D)`` tensor of L2-normalized
            features where ``P`` is the prompt count for that label and
            ``D`` is CLIP's embedding dimension (512 for ViT-B/32).
        """
        import torch

        model, _, tokenizer = self._load_model()
        encoded: dict[str, object] = {}
        for label, prompts in label_prompts.items():
            tokens = tokenizer(prompts)
            with torch.no_grad():
                features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            encoded[label] = features
        return encoded

    def _encode_images_batch(self, frame_paths: list[Path]) -> "object":
        """Load + preprocess + encode a list of frames in a single CLIP forward pass.

        Returns an ``(N, D)`` L2-normalized image-feature tensor where
        ``N == len(frame_paths)`` and ``D`` is CLIP's embedding dimension.
        Raises ``OSError`` / ``RuntimeError`` / ``ValueError`` to surface
        per-batch failures so the caller can fall back to a per-frame
        retry path. Decoding is sequential because PIL is not thread-safe
        with shared file descriptors; the speedup comes from a single
        batched ``encode_image`` call.
        """
        import torch
        from PIL import Image

        model, preprocess, _ = self._load_model()
        images = [
            preprocess(Image.open(fp).convert("RGB")) for fp in frame_paths
        ]
        batch = torch.stack(images, dim=0)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    @staticmethod
    def _zero_shot_multi_batch(
        image_features: "object",
        label_text_features: dict[str, "object"],
        threshold: float,
    ) -> list[list[str]]:
        """Per-frame zero-shot classification against pre-encoded text features.

        Args:
            image_features: ``(B, D)`` L2-normalized image features.
            label_text_features: dict mapping label to ``(P, D)`` L2-
                normalized text features (output of
                ``_encode_label_prompts``).
            threshold: cosine-similarity threshold; a label is matched
                for a given frame iff its max-over-prompts similarity
                meets or exceeds this value.

        Returns:
            A list of length ``B``; the i-th entry is the list of
            matched labels for the i-th frame in ``image_features``.
        """
        batch_size = image_features.shape[0]
        matched: list[list[str]] = [[] for _ in range(batch_size)]
        for label, text_features in label_text_features.items():
            # (B, D) @ (D, P) → (B, P)
            similarity = image_features @ text_features.T
            # Per-frame max across prompts → (B,)
            max_sim_per_frame = similarity.max(dim=-1).values
            for b in range(batch_size):
                if max_sim_per_frame[b].item() >= threshold:
                    matched[b].append(label)
        return matched

    def tag_frames(
        self,
        frame_paths: list[Path],
        brand: str = "",
        category: str = "",
        fps: float = 1.0,
        scene_threshold: float = 0.24,
        emotion_threshold: float = 0.25,
        object_threshold: float = 0.27,
    ) -> list[FrameVisualTags]:
        """Run zero-shot classification on every supplied frame.

        Frame-level entry point: every 1 fps frame produced by the
        normalizer is classified, rather than a single representative
        frame per shot. The pipeline relies on this so that signals
        concentrated at the opening / closing of an ad (logo cards,
        CTA cards) are not hidden by middle-frame sampling.

        Frames are sorted by parsed frame index, batched into chunks of
        ``self.batch_size``, and encoded in a single CLIP forward pass
        per batch. Text prompts (scene / emotion / object) are encoded
        exactly once at entry and reused across all batches. On a
        per-batch failure the method falls back to ``_classify_frame``
        for each frame in that batch so a single corrupted PNG cannot
        wipe out the whole video's results.

        Args:
            frame_paths: paths to PNG frames (typically 1 fps output of
                ``VideoNormalizer``). Order is non-strict; the method
                sorts internally by parsed frame index.
            brand / category: optional metadata that augments object
                prompts (see ``build_object_prompts``).
            fps: frame rate used to derive ``time_sec`` from
                ``frame_idx``. Defaults to 1.0 to match the pipeline's
                normalizer setting.
            scene_threshold / emotion_threshold / object_threshold:
                cosine-similarity thresholds, identical defaults to
                ``_classify_frame`` for behavioural compatibility.

        Returns:
            One ``FrameVisualTags`` per input frame, sorted by
            ``frame_idx`` ascending. Frames whose file is missing or
            whose classification fails are emitted with all flags
            ``False`` and empty tag lists (so downstream
            ``time_bins`` construction is index-stable).
        """
        if not frame_paths:
            return []

        # Sort by parsed frame index so the returned list aligns with
        # ``time_bins`` order (bin_id == frame_idx - 1 when fps=1).
        ordered: list[tuple[int, Path]] = []
        for fp in frame_paths:
            try:
                ordered.append((self._parse_frame_idx(fp), fp))
            except VisualTaggerError as exc:
                logger.warning("tag_frames: %s; skipping", exc)
        ordered.sort(key=lambda item: item[0])
        if not ordered:
            return []

        object_prompts = build_object_prompts(brand=brand, category=category)
        # Risk-1 mitigation: encode every text prompt set exactly once,
        # before any image inference. The returned tensors are reused
        # across every image batch below, so per-call tokenizer state
        # cannot leak between batches.
        try:
            scene_text = self._encode_label_prompts(_SCENE_PROMPTS)
            emotion_text = self._encode_label_prompts(_EMOTION_PROMPTS)
            object_text = self._encode_label_prompts(object_prompts)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error(
                "tag_frames: text-prompt encoding failed (%s); "
                "returning empty tags for all %d frames.",
                exc, len(ordered),
            )
            return [
                FrameVisualTags(frame_idx=idx, time_sec=(idx - 1) / fps)
                for idx, _ in ordered
            ]

        results: list[FrameVisualTags] = []
        for batch_start in range(0, len(ordered), self.batch_size):
            chunk = ordered[batch_start:batch_start + self.batch_size]
            chunk_paths = [fp for _, fp in chunk]
            try:
                image_features = self._encode_images_batch(chunk_paths)
            except (OSError, RuntimeError, ValueError) as exc:
                # Per-batch failure: fall back to single-frame
                # ``_classify_frame`` so one corrupted PNG cannot
                # zero out an otherwise-fine batch.
                logger.warning(
                    "tag_frames: batch [%d..%d) failed (%s); "
                    "falling back per-frame.",
                    batch_start, batch_start + len(chunk), exc,
                )
                for idx, fp in chunk:
                    results.append(
                        self._tag_single_frame_fallback(
                            idx, fp, fps, object_prompts,
                            scene_threshold, emotion_threshold, object_threshold,
                        )
                    )
                continue

            scenes_per = self._zero_shot_multi_batch(
                image_features, scene_text, scene_threshold
            )
            emotions_per = self._zero_shot_multi_batch(
                image_features, emotion_text, emotion_threshold
            )
            objects_per = self._zero_shot_multi_batch(
                image_features, object_text, object_threshold
            )
            for i, (idx, _) in enumerate(chunk):
                results.append(
                    FrameVisualTags(
                        frame_idx=idx,
                        time_sec=(idx - 1) / fps,
                        logo_visible="logo" in objects_per[i],
                        product_visible="product" in objects_per[i],
                        cta_visible="cta_button" in objects_per[i],
                        emotion_tags=sorted(emotions_per[i]),
                        scene_tags=sorted(scenes_per[i]),
                    )
                )
        return results

    def _tag_single_frame_fallback(
        self,
        frame_idx: int,
        frame_path: Path,
        fps: float,
        object_prompts: dict[str, list[str]],
        scene_threshold: float,
        emotion_threshold: float,
        object_threshold: float,
    ) -> FrameVisualTags:
        """Per-frame fallback when batch encoding fails.

        Reuses the single-image ``_classify_frame`` path so the
        fallback cannot diverge in its scoring logic from the batched
        path. On a single-frame failure the method emits a no-content
        ``FrameVisualTags`` so the result list stays index-stable
        with the caller's frame list.
        """
        try:
            tags = self._classify_frame(
                frame_path, object_prompts,
                scene_threshold=scene_threshold,
                emotion_threshold=emotion_threshold,
                object_threshold=object_threshold,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "tag_frames: per-frame fallback failed for %s: %s",
                frame_path, exc,
            )
            return FrameVisualTags(
                frame_idx=frame_idx, time_sec=(frame_idx - 1) / fps,
            )
        return FrameVisualTags(
            frame_idx=frame_idx,
            time_sec=(frame_idx - 1) / fps,
            logo_visible="logo" in tags.get("objects", []),
            product_visible="product" in tags.get("objects", []),
            cta_visible="cta_button" in tags.get("objects", []),
            emotion_tags=sorted(tags.get("emotions", [])),
            scene_tags=sorted(tags.get("scenes", [])),
        )
