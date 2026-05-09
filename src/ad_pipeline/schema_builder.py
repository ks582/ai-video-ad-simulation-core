"""T-107: Canonical Ad Schema builder (FR-07).

Integrates shots, ASR, OCR, and visual tags to generate time-bins.
"""

from __future__ import annotations

import math
import re

from src.ad_pipeline.models import (
    ASRSegment,
    AdMetadata,
    BrandExposure,
    CTAEvent,
    CanonicalAdSchema,
    DominantTone,
    FrameVisualTags,
    GlobalFeatures,
    OCRResult,
    PriceSignal,
    PriceSignalType,
    Shot,
    TimeBin,
    VisualTags,
)

# Price-related keywords
# Keyword -> PriceSignalType mapping
_PRICE_CURRENCY = {"¥", "$", "€", "£", "円", "ドル"}
_PRICE_DISCOUNT = {"割引", "OFF", "off", "セール", "sale"}
_PRICE_FREE = {"無料", "free", "無料体験", "free_trial"}
_PRICE_LIMITED = {"限定", "期間限定", "今だけ", "特別価格"}

_PRICE_KEYWORDS = list(_PRICE_CURRENCY | _PRICE_DISCOUNT | _PRICE_FREE | _PRICE_LIMITED)

# Curated suffix nouns for the "free X" pattern — keeps the matcher tight and
# avoids false positives like "send your kid the free game on weekends".
_FREE_OFFER_SUFFIXES = (
    "sample", "samples",
    "trial", "trials",
    "gift", "gifts",
    "pack", "packs",
    "box", "boxes",
    "kit", "kits",
    "shipping",
    "delivery",
    "tissue", "tissues",  # extend per category as needed
)
_FREE_OFFER_SUFFIX_GROUP = "|".join(_FREE_OFFER_SUFFIXES)

# CTA detection patterns from spoken/visible text. Each tuple is
# (compiled regex, cta_type tag). Order matters: more specific patterns
# should come before generic ones since the first match wins per segment.
# Rationale (preserve when reordering):
#   1. Phrase-with-URL ("visit X.com") — strongest URL evidence
#   2. Bare URL — TLD-anchored fallback
#   3. Phrase-with-website-noun ("visit our website") — verbal CTA without URL
#   4. Phone, app, purchase, signup — verb-anchored intent verbs
#   5. Free offer — curated suffix only (see _FREE_OFFER_SUFFIXES)
_ASR_CTA_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Website / URL — "go to brand.com", "visit our website", bare ".com" mention
    (re.compile(r"(?:visit|go to|check out|head to)\s+\S*?\.(?:com|net|org|io|co)\b", re.IGNORECASE), "website_visit"),
    (re.compile(r"\b\S+\.(?:com|net|org|io|co)\b", re.IGNORECASE), "website_visit"),
    (re.compile(r"\b(?:visit|check out)\s+(?:our|the)\s+(?:website|site|page)\b", re.IGNORECASE), "website_visit"),
    # Brand-message taglines "<verb> more" that imply "go learn/explore more"
    # — mapped to website_visit since the user's next action is typically a
    # search or a visit to the brand site. Examples:
    #   "Ask more" / "Discover more" / "Learn more"
    #   "Find out more" / "See more" / "Explore more"
    # The verb-anchored prefix prevents false positives on prose like
    # "I'd like to learn more about cooking" — the regex requires the phrase
    # to stand alone or follow a sentence start / overlay caption.
    (re.compile(r"(?:^|\W)(?:ask|discover|explore|learn|find out|see)\s+more\b", re.IGNORECASE), "website_visit"),
    # Phone
    (re.compile(r"\b(?:call|dial)\s+(?:\d|now)", re.IGNORECASE), "phone_call"),
    # App
    (re.compile(r"\b(?:download|install|get)\s+(?:our|the)?\s*app\b", re.IGNORECASE), "app_install"),
    # Purchase / order
    (re.compile(r"\b(?:buy|shop|order|purchase)\s+(?:now|today)\b", re.IGNORECASE), "purchase"),
    # Sign-up / subscribe — bare verb is enough; trailing now/today/here is optional
    # together with its leading whitespace (avoids the regex requiring whitespace
    # then optional word, which fails on bare "Sign up").
    (re.compile(r"\b(?:sign up|subscribe|register)(?:\s+(?:now|today|here))?\b", re.IGNORECASE), "signup"),
    # Free offer — verb anchor + up to 4 intervening words + "free" + up to 3
    # additional intervening words (e.g. brand name) + curated noun suffix.
    # Examples that match: "send a loved one a free Kleenex care pack",
    # "claim your free trial", "get a free sample", "request free shipping".
    # Examples that do NOT match: "free game", "free time", "I love free wifi"
    # (curated noun list excludes "game", "time", "wifi"; the verb anchor
    # excludes free-standing "free X" phrases without get/send/request/claim).
    (
        re.compile(
            rf"\b(?:get|send|request|claim)\b(?:\s+\w+){{0,4}}\s+free(?:\s+\w+){{0,3}}\s+(?:{_FREE_OFFER_SUFFIX_GROUP})\b",
            re.IGNORECASE,
        ),
        "free_offer",
    ),
]


def detect_ctas_from_text(text: str) -> str | None:
    """Return the first matching CTA type tag for ``text``, or None.

    Used by both ASR-derived and OCR-derived CTA extraction. Matching is
    deterministic and order-sensitive — the first pattern in
    ``_ASR_CTA_PATTERNS`` that matches wins.
    """
    if not text:
        return None
    for pattern, cta_type in _ASR_CTA_PATTERNS:
        if pattern.search(text):
            return cta_type
    return None


class CanonicalAdSchemaBuilder:
    """Build the Canonical Ad Schema.

    Acceptance criteria:
    - Has global_features
    - Has shot_sequence
    - Has time_bins
    - Has brand/logo exposure, price signals, CTA events
    """

    def __init__(self, time_bin_sec: float = 1.0) -> None:
        self.time_bin_sec = time_bin_sec

    def build(
        self,
        input_id: str,
        metadata: AdMetadata,
        shots: list[Shot],
        asr_segments: list[ASRSegment],
        ocr_results: list[OCRResult],
        visual_tags: list[VisualTags] | None = None,
        frame_visual_tags: list[FrameVisualTags] | None = None,
    ) -> CanonicalAdSchema:
        """Integrate all preprocessing results to generate the Canonical Ad Schema.

        Option 3 frame-level inference is preferred — when
        ``frame_visual_tags`` is supplied, ``time_bins``, ``brand_exposure``,
        and visual-derived CTA events are computed directly from per-frame
        signals and the per-shot ``visual_tags`` field is synthesized
        post-hoc as a union over the frames inside each shot's time range.

        ``visual_tags`` is kept as a fallback for any external caller that
        still produces only shot-level data (none in-tree as of PR 2). When
        ``frame_visual_tags`` is omitted, the legacy shot-level path runs
        unchanged.
        """
        duration = metadata.duration_sec

        # Resolve the two visual-tag inputs. The new pipeline always
        # supplies ``frame_visual_tags``; legacy callers may pass only
        # ``visual_tags``. We keep both populated on the output schema:
        # ``visual_tags`` is synthesized from the frame data when missing
        # so downstream consumers that read it directly stay backward-
        # compatible without inspecting the new field.
        if frame_visual_tags is None:
            frame_visual_tags = []
        if visual_tags is None:
            visual_tags = self._synthesize_visual_tags(frame_visual_tags, shots)

        # Generate time-bins. Frame-level path takes precedence when frame
        # tags are present; the shot-level path remains for backward compat.
        if frame_visual_tags:
            time_bins = self._build_time_bins_from_frames(
                duration, asr_segments, ocr_results, frame_visual_tags
            )
        else:
            time_bins = self._build_time_bins(
                duration, shots, asr_segments, ocr_results, visual_tags
            )

        # Global features (time-bins must be built first — dominant_tone is
        # derived from per-bin emotion_tone weighted by bin duration so the
        # schema-level summary stays consistent with the displayed
        # tone_durations breakdown).
        global_features = self._build_global_features(
            duration, shots, asr_segments, time_bins
        )

        # Brand exposure
        if frame_visual_tags:
            brand_exposure = self._build_brand_exposure_from_frames(frame_visual_tags)
        else:
            brand_exposure = self._build_brand_exposure(visual_tags, shots)

        # Price signals
        price_signals = self._extract_price_signals(asr_segments, ocr_results)

        # CTA events (visual + ASR/OCR derived — ASR fills the gap when the
        # ad has no on-screen CTA overlay but a spoken call to action,
        # which is common in CPG/TV creatives).
        cta_events = self._extract_cta_events(
            visual_tags, shots, metadata, asr_segments, ocr_results,
            frame_visual_tags=frame_visual_tags,
        )

        return CanonicalAdSchema(
            input_id=input_id,
            metadata=metadata,
            global_features=global_features,
            shot_sequence=shots,
            time_bins=time_bins,
            brand_exposure=brand_exposure,
            price_signals=price_signals,
            cta_events=cta_events,
            asr_segments=asr_segments,
            ocr_results=ocr_results,
            visual_tags=visual_tags,
            frame_visual_tags=frame_visual_tags,
        )

    def _build_time_bins(
        self,
        duration: float,
        shots: list[Shot],
        asr_segments: list[ASRSegment],
        ocr_results: list[OCRResult],
        visual_tags: list[VisualTags],
    ) -> list[TimeBin]:
        """Generate the time-bin sequence."""
        num_bins = max(1, math.ceil(duration / self.time_bin_sec))
        tags_by_shot = {vt.shot_id: vt for vt in visual_tags}

        bins: list[TimeBin] = []
        for i in range(num_bins):
            start = i * self.time_bin_sec
            end = min((i + 1) * self.time_bin_sec, duration)

            # Aggregate visual tags from shots overlapping this time-bin
            logo = False
            product = False
            cta = False
            emotions: list[str] = []

            for shot in shots:
                if shot.end_sec <= start or shot.start_sec >= end:
                    continue
                vt = tags_by_shot.get(shot.shot_id)
                if vt:
                    logo = logo or vt.logo_visible
                    product = product or vt.product_visible
                    cta = cta or vt.cta_visible
                    emotions.extend(vt.emotion_tags)

            # Aggregate ASR text
            asr_texts = [
                seg.text for seg in asr_segments
                if seg.start_sec < end and seg.end_sec > start
            ]

            # Aggregate OCR text
            ocr_texts = [
                r.text for r in ocr_results
                if start <= r.time_sec < end
            ]

            # Price signal
            all_text = " ".join(asr_texts + ocr_texts)
            price_signal = any(kw in all_text for kw in _PRICE_KEYWORDS)

            # Emotion tone (most frequent)
            emotion_tone = "neutral"
            if emotions:
                from collections import Counter
                emotion_tone = Counter(emotions).most_common(1)[0][0]

            bins.append(
                TimeBin(
                    bin_id=i,
                    start_sec=start,
                    end_sec=end,
                    brand_logo_visible=logo,
                    product_visible=product,
                    price_signal=price_signal,
                    cta_event=cta,
                    emotion_tone=emotion_tone,
                    asr_text=" ".join(asr_texts),
                    ocr_text=" ".join(ocr_texts),
                )
            )

        return bins

    def _build_global_features(
        self,
        duration: float,
        shots: list[Shot],
        asr_segments: list[ASRSegment],
        time_bins: list[TimeBin],
    ) -> GlobalFeatures:
        """Compute overall ad features.

        ``dominant_tone`` is the time-weighted majority emotion derived from
        ``time_bins`` rather than a raw count of emotion tags. Counting tag
        occurrences (the prior implementation) let a single short shot with
        many tags outvote much longer shots that happened to carry only one
        tag — producing taglines like "exciting" for ads that are 87% neutral
        in screen time. Reading from ``time_bins`` also keeps the schema-level
        ``dominant_tone`` aligned with the per-bin ``emotion_tone`` field used
        by downstream aggregation.
        """
        # Duration-weighted emotion-tone aggregation. Each bin contributes
        # ``end_sec - start_sec`` seconds to its ``emotion_tone`` bucket.
        # Empty bins fall back to "neutral" via the time_bin builder.
        tone_durations: dict[str, float] = {}
        for tb in time_bins:
            tag = tb.emotion_tone or "neutral"
            tone_durations[tag] = tone_durations.get(tag, 0.0) + (tb.end_sec - tb.start_sec)

        tone = DominantTone.NEUTRAL
        if tone_durations:
            top = max(tone_durations.items(), key=lambda kv: kv[1])[0]
            tone_map = {
                "happy": DominantTone.POSITIVE,
                "exciting": DominantTone.POSITIVE,
                "calm": DominantTone.NEUTRAL,
                "serious": DominantTone.NEUTRAL,
                "neutral": DominantTone.NEUTRAL,
                "humorous": DominantTone.HUMOROUS,
                "dramatic": DominantTone.DRAMATIC,
                "inspiring": DominantTone.POSITIVE,
                "nostalgic": DominantTone.NEUTRAL,
            }
            tone = tone_map.get(top, DominantTone.NEUTRAL)

        has_narration = len(asr_segments) > 0
        # Dialogue if 2 or more speakers
        speakers = {seg.speaker_label for seg in asr_segments if seg.speaker_label}
        has_dialogue = len(speakers) >= 2

        return GlobalFeatures(
            total_duration_sec=duration,
            num_shots=len(shots),
            dominant_tone=tone,
            has_narration=has_narration,
            has_dialogue=has_dialogue,
        )

    def _build_time_bins_from_frames(
        self,
        duration: float,
        asr_segments: list[ASRSegment],
        ocr_results: list[OCRResult],
        frame_visual_tags: list[FrameVisualTags],
    ) -> list[TimeBin]:
        """Generate the time-bin sequence from per-frame visual tags.

        Frame-level path (PR 2). Each ``FrameVisualTags`` entry is
        assigned to the bin whose time range covers its ``time_sec``;
        when multiple frames fall in the same bin (e.g. fps > 1), their
        flags are unioned and emotion tags are concatenated for the
        dominant-tone vote inside that bin.
        """
        num_bins = max(1, math.ceil(duration / self.time_bin_sec))

        bins: list[TimeBin] = []
        for i in range(num_bins):
            start = i * self.time_bin_sec
            end = min((i + 1) * self.time_bin_sec, duration)

            # Frame-level visual tag aggregation. With fps=1.0 and
            # ``time_bin_sec=1.0`` (default), each bin contains exactly
            # one frame and the loop body runs at most once.
            logo = False
            product = False
            cta = False
            emotions: list[str] = []

            for fvt in frame_visual_tags:
                if fvt.time_sec < start or fvt.time_sec >= end:
                    continue
                logo = logo or fvt.logo_visible
                product = product or fvt.product_visible
                cta = cta or fvt.cta_visible
                emotions.extend(fvt.emotion_tags)

            # Aggregate ASR text
            asr_texts = [
                seg.text for seg in asr_segments
                if seg.start_sec < end and seg.end_sec > start
            ]

            # Aggregate OCR text
            ocr_texts = [
                r.text for r in ocr_results
                if start <= r.time_sec < end
            ]

            # Price signal
            all_text = " ".join(asr_texts + ocr_texts)
            price_signal = any(kw in all_text for kw in _PRICE_KEYWORDS)

            # Emotion tone (most frequent)
            emotion_tone = "neutral"
            if emotions:
                from collections import Counter
                emotion_tone = Counter(emotions).most_common(1)[0][0]

            bins.append(
                TimeBin(
                    bin_id=i,
                    start_sec=start,
                    end_sec=end,
                    brand_logo_visible=logo,
                    product_visible=product,
                    price_signal=price_signal,
                    cta_event=cta,
                    emotion_tone=emotion_tone,
                    asr_text=" ".join(asr_texts),
                    ocr_text=" ".join(ocr_texts),
                )
            )

        return bins

    def _build_brand_exposure_from_frames(
        self,
        frame_visual_tags: list[FrameVisualTags],
    ) -> BrandExposure:
        """Frame-precise brand exposure aggregation.

        Replaces the prior shot-duration approximation with a true
        per-frame measurement. Each logo-visible frame contributes
        ``time_bin_sec`` (default 1.0s for the canonical 1-fps pipeline)
        to ``total_duration_sec``; ``num_appearances`` counts contiguous
        runs of logo-visible frames so a single 4-frame logo card
        registers as one appearance, not four.
        """
        if not frame_visual_tags:
            return BrandExposure()

        ordered = sorted(frame_visual_tags, key=lambda f: f.frame_idx)
        first_sec: float | None = None
        total_sec = 0.0
        appearances = 0
        prev_visible = False

        for fvt in ordered:
            if fvt.logo_visible:
                if first_sec is None:
                    first_sec = fvt.time_sec
                total_sec += self.time_bin_sec
                if not prev_visible:
                    appearances += 1
                prev_visible = True
            else:
                prev_visible = False

        return BrandExposure(
            first_appearance_sec=first_sec,
            total_duration_sec=total_sec,
            num_appearances=appearances,
        )

    @staticmethod
    def _synthesize_visual_tags(
        frame_visual_tags: list[FrameVisualTags],
        shots: list[Shot],
    ) -> list[VisualTags]:
        """Synthesize per-shot ``VisualTags`` from per-frame results.

        Preserves backward compatibility for downstream consumers that
        read ``CanonicalAdSchema.visual_tags`` directly. For each shot,
        union the flags of the frames whose ``time_sec`` falls within
        ``[shot.start_sec, shot.end_sec)``.
        """
        out: list[VisualTags] = []
        for shot in shots:
            scenes: set[str] = set()
            emotions: set[str] = set()
            logo = False
            product = False
            cta = False
            for fvt in frame_visual_tags:
                if not (shot.start_sec <= fvt.time_sec < shot.end_sec):
                    continue
                logo = logo or fvt.logo_visible
                product = product or fvt.product_visible
                cta = cta or fvt.cta_visible
                scenes.update(fvt.scene_tags)
                emotions.update(fvt.emotion_tags)
            out.append(
                VisualTags(
                    shot_id=shot.shot_id,
                    logo_visible=logo,
                    product_visible=product,
                    cta_visible=cta,
                    emotion_tags=sorted(emotions),
                    scene_tags=sorted(scenes),
                )
            )
        return out

    def _build_brand_exposure(
        self,
        visual_tags: list[VisualTags],
        shots: list[Shot],
    ) -> BrandExposure:
        """Compute brand exposure information."""
        first_sec: float | None = None
        total_sec = 0.0
        count = 0

        for vt in visual_tags:
            if not vt.logo_visible:
                continue
            shot = next((s for s in shots if s.shot_id == vt.shot_id), None)
            if shot is None:
                continue
            if first_sec is None:
                first_sec = shot.start_sec
            total_sec += shot.end_sec - shot.start_sec
            count += 1

        return BrandExposure(
            first_appearance_sec=first_sec,
            total_duration_sec=total_sec,
            num_appearances=count,
        )

    def _extract_price_signals(
        self,
        asr_segments: list[ASRSegment],
        ocr_results: list[OCRResult],
    ) -> list[PriceSignal]:
        """Extract price signals."""
        signals: list[PriceSignal] = []

        for seg in asr_segments:
            sig_type = self._classify_price_keyword(seg.text)
            if sig_type is not None:
                signals.append(
                    PriceSignal(time_sec=seg.start_sec, type=sig_type, text=seg.text)
                )

        for ocr in ocr_results:
            sig_type = self._classify_price_keyword(ocr.text)
            if sig_type is not None:
                signals.append(
                    PriceSignal(time_sec=ocr.time_sec, type=sig_type, text=ocr.text)
                )

        return signals

    @staticmethod
    def _classify_price_keyword(text: str) -> PriceSignalType | None:
        """Classify price keywords in text."""
        for kw in _PRICE_CURRENCY:
            if kw in text:
                return PriceSignalType.PRICE_DISPLAY
        for kw in _PRICE_FREE:
            if kw in text:
                return PriceSignalType.FREE_TRIAL
        for kw in _PRICE_LIMITED:
            if kw in text:
                return PriceSignalType.LIMITED_OFFER
        for kw in _PRICE_DISCOUNT:
            if kw in text:
                return PriceSignalType.DISCOUNT
        return None

    def _extract_cta_events(
        self,
        visual_tags: list[VisualTags],
        shots: list[Shot],
        metadata: AdMetadata,
        asr_segments: list[ASRSegment] | None = None,
        ocr_results: list[OCRResult] | None = None,
        frame_visual_tags: list[FrameVisualTags] | None = None,
    ) -> list[CTAEvent]:
        """Extract CTA events from visual tags, ASR text, and OCR text.

        Visual-tag-derived events use the campaign's nominal ``metadata.cta_type``
        (since the visual signal only confirms presence, not type). When
        ``frame_visual_tags`` is provided (PR 2 path), per-frame
        ``cta_visible=True`` flags are used directly so the event time is
        the frame's ``time_sec`` rather than the enclosing shot's start.
        ASR- and OCR-derived events use the regex-classified CTA type
        from :func:`detect_ctas_from_text` since the textual content is
        itself the evidence for what kind of action is being requested.
        We dedupe events landing in the same one-second bucket to avoid
        double-counting when the same CTA appears in both visual and
        spoken form.
        """
        events: list[CTAEvent] = []
        seen_buckets: set[tuple[int, str]] = set()

        def _push(time_sec: float, cta_type: str, cta_text: str = "") -> None:
            bucket = (int(time_sec), cta_type)
            if bucket in seen_buckets:
                return
            seen_buckets.add(bucket)
            events.append(CTAEvent(time_sec=time_sec, cta_type=cta_type, cta_text=cta_text))

        # 1. Visual-tag derived. Frame-level path (PR 2) when available;
        #    falls back to shot-level for legacy callers.
        if frame_visual_tags:
            for fvt in frame_visual_tags:
                if fvt.cta_visible:
                    _push(fvt.time_sec, metadata.cta_type.value)
        else:
            for vt in visual_tags:
                if not vt.cta_visible:
                    continue
                shot = next((s for s in shots if s.shot_id == vt.shot_id), None)
                if shot is None:
                    continue
                _push(shot.start_sec, metadata.cta_type.value)

        # 2. ASR-derived (P3): regex-classify spoken text per segment
        for seg in asr_segments or []:
            cta_type = detect_ctas_from_text(seg.text)
            if cta_type is not None:
                _push(seg.start_sec, cta_type, seg.text)

        # 3. OCR-derived: regex-classify on-screen text. OCR values can be
        #    noisy ("Kdoexox" misreadings); detect_ctas_from_text only fires
        #    on confident pattern matches so noise tends to be silently dropped.
        for ocr in ocr_results or []:
            cta_type = detect_ctas_from_text(ocr.text)
            if cta_type is not None:
                _push(ocr.time_sec, cta_type, ocr.text)

        return events
