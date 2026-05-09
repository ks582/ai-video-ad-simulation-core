# AI Video Ad Simulation: Synthetic Pretest System

A synthetic pretest platform that simulates video ad delivery to AI agents (synthetic viewers) using Chain of Thought (CoT) reasoning to model how target audiences process and respond to ad exposure — enabling advertisers to estimate brand lift metrics before real-world deployment.

**License:** [BSL 1.1](LICENSE.md) (converts to Apache 2.0 after 5 years)

---

## Overview

Instead of buying media inventory and measuring real audiences after the fact, this system delivers video ads to **AI agents acting as synthetic viewers** — and measures the effect computationally, before any real budget is spent.

Each AI agent is initialized with a structured persona (demographics, brand priors, behavioral propensities) sampled from public population statistics. The agent then processes the ad through a multimodal LLM and its internal state is updated to reflect exposure. Comparing agents who saw the ad (Treatment) against statistically identical agents who did not (Control) yields a causal estimate of ad effectiveness.

### Pipeline Overview

```mermaid
flowchart TD
    A([Video Input]) --> B[Tier 1: Ad Pipeline\nVideo → Canonical Ad Schema]
    B --> C[Tier 2: Persona Generation\nStatistical sampling → Twin pairs]
    C --> D[Tier 3: Simulation Environment\nSession context + ad insertion]
    D --> E{Twin-World Fork}

    E -->|Treatment| F1[Call 0-A: Segment Baseline]
    F1 --> F2[Call 0-B: Individual Baseline]
    F2 --> F3[Call 1: Viewing Narrative\nAgent watches the ad]
    F3 --> F4[Call 2: Delta Estimation\nMetric shifts from exposure]

    E -->|Control| G1[Call 0-A: Segment Baseline]
    G1 --> G2[Call 0-B: Individual Baseline]
    G2 --> G3[Control Call 1\nNo-ad session state]
    G3 --> G4[Control Call 2\nNear-zero drift estimation]

    F4 --> H[Tier 5: Statistical Analysis\nPaired difference estimation + CI]
    G4 --> H

    H --> I[Tier 6: Diagnosis\nRoot cause attribution]
    I --> J([Tier 7: Integration Outputs\nSchemas + downstream report contracts])
```

<!-- Static fallback: ![Pipeline Overview](docs/images/pipeline_overview.png) -->

The LLM-Agent Simulation Loop shares baseline estimation across both worlds (Call 0-A/0-B), then runs symmetric CoT-based measurement flows: Treatment sees the ad and estimates exposure-driven response deltas; Control receives no ad stimulus and estimates only counterfactual no-ad drift. The only causal difference between the two paths is ad exposure.

### Metrics Estimated

The paired comparison produces lift estimates (with confidence intervals) across seven brand-health dimensions, ordered along the brand funnel:

- **Upper funnel (awareness):** brand awareness lift, ad recall lift
- **Mid funnel (attitudes):** message association lift, favorability lift (unipolar [0, 1] — `0` means no positive feeling, `1` means maximally favorable)
- **Lower funnel (action):** consideration lift, purchase intent lift, search intent lift

For every metric the report also surfaces **Headroom Lift** (`Abs Lift / (1 − Control)` — the share of the remaining baseline headroom captured by the ad; suppressed when the control rate exceeds 0.90 because the denominator becomes unstable) and a deterministic 4-paragraph synthesis above the metrics table that summarises the funnel-level story without using an LLM. The system also outputs **root cause analysis** and **actionable improvement hypotheses** based on ad structure and agent feedback.

---

## Design Philosophy

### The Problem: Ad Performance Is Empirical by Default

Traditionally, the only way to know whether a video advertisement will work is to run it — deploy real media budget, reach real audiences, and measure what happens afterward. This creates an expensive feedback loop: creative teams produce an ad, planners buy inventory, and only after real-world exposure do brands discover whether the execution moved the metrics they cared about. Each iteration cycle costs money, and the learnings arrive too late to inform the current campaign.

### Synthetic Pretesting with AI Agents

This project takes a different approach: instead of exposing ads to real audiences, the system initializes **AI agents as synthetic viewers**. Each agent carries a set of demographic and psychographic state variables — drawn from publicly available population statistics (ACS, CPS, CEX) — that represent a member of the intended target segment. The agent then "watches" the ad through a multimodal model and its internal state variables are updated to reflect the exposure.

The essence of this architecture is **making AI agents behave like humans**. Personas are modeled as structured state variables, not open-ended personality narratives. This is an intentional choice: state variables are observable, comparable, and constrained, which makes their changes after ad exposure interpretable as measurable deltas rather than free-form impressions.

### Canonical Causal Estimation Pipeline

Ad effectiveness is not simply "how did you feel after watching?" — it is the **difference** between a world where you watched the ad and a world where you did not. The pipeline operationalizes this through shared baseline calls plus arm-specific LLM measurement calls:

1. **Call 0-A (Segment Baseline):** Estimates 7 brand-lift metrics at the segment level — what would a typical member of this demographic cell score *without* the ad? This anchors all subsequent per-persona estimates.
2. **Call 0-B (Individual Baseline):** Adjusts the segment baseline for each persona's specific attributes (ad tolerance, brand familiarity, price sensitivity, etc.), producing per-persona prior distributions.
3. **Treatment Call 1/2:** Treatment agents watch the ad, produce a first-person viewing narrative, and then estimate exposure-driven deltas from their individual baselines.
4. **Control Call 1/2:** Control agents receive no ad stimulus, estimate normal no-ad session state, and then estimate small counterfactual drift around the same individual baselines.

**Treatment vs. Control:** The current implementation shares Call 0-A and Call 0-B across each twin pair, then runs two arm-specific calls. Treatment Call 1 builds the ad-viewing narrative and Treatment Call 2 estimates metric deltas from that narrative. Control Call 1 estimates no-ad baseline session state and Control Call 2 estimates small no-ad drift around the same individual baselines. Earlier Beta-sampling control logic has been superseded by this symmetric LLM control path.

### Mathematical View

At a high level, the system treats an ad video as a structured stimulus:

```math
A = \Phi(V, m) = \{x_t\}_{t=1}^{T}
```

where \(V\) is the input video, \(m\) is campaign metadata, and each \(x_t\)
is a time-indexed representation of what the viewer can observe at second
\(t\): visual features, spoken text, on-screen text, brand exposure, product
presence, and CTA signals.

Each synthetic viewer is evaluated through a paired twin-world design:

```math
P_i^T = P_i^C
```

The treatment and control twins share the same persona state. The only
intervention is whether the ad stimulus is present.

The system estimates a baseline and then compares treatment and control
outcomes:

```math
b_i = f_0(P_i)
```

```math
Y_i^T = f_T(P_i, A, b_i), \quad
Y_i^C = f_C(P_i, b_i)
```

For each brand-lift metric \(k\), the paired difference is:

```math
d_{ik} = Y_{ik}^{T} - Y_{ik}^{C}
```

and the estimated lift is:

```math
\hat{\tau}_k = \frac{1}{N}\sum_{i=1}^{N} d_{ik}
```

Headroom Lift expresses how much of the remaining control-world headroom the
ad captures:

```math
\mathrm{HeadroomLift}_k =
\frac{\hat{\tau}_k}{1 - \bar{Y}_k^C}
```

Reported estimates may pass through an internal null-stimulus bias correction
layer to reduce systematic LLM response drift. The calibration procedure is
intentionally not described in this public README.

### Known Limitations

Two structural limitations are worth stating explicitly.

**No real-world calibration data.** The system operates without ground truth — there is no real ad delivery data to validate against. Internal bias correction may be applied to reduce systematic LLM response drift, but the surviving estimates are still anchored to LLM judgments rather than observed campaign outcomes. They are directionally useful for relative comparisons (ad variant A vs. B) but are not numerically calibrated against real engagement. The architecture currently has no feedback path from simulation results back to the baseline estimators or prompt tuning.

**Chain of Thought self-reinforcement.** Because Call 2 conditions on Call 1's narrative output, the same model's prior beliefs propagate forward through the pipeline. If the LLM consistently over- or under-represents certain response patterns in viewing narratives, those biases compound into the final delta estimates. The system mitigates this with deterministic funnel coherence clamping (hard constraints on metric ordering), but this corrects logical contradictions — not magnitude errors. Addressing this limitation is an open area for future work, including ensemble sampling across models and external calibration anchors from real brand lift studies.

**Hand-tuned persona affinity constants.** The persona affinity presets and the relevance/boost coefficients used during simulation are defined by manual heuristic judgment (each persona type carries 3–6 weighted vertical interests; relevance decay and behavior-boost magnitudes are fixed constants). These constants have not been fit to real ad delivery logs or validated against observed engagement rates. They govern directional relative comparisons between ad variants but carry no quantitative calibration. Future versions may derive these values from real delivery logs or from LLM-based persona-ad match inference.

---

## Architecture

The pipeline consists of 7 tiers:

```
Video Input
  -> [Tier 1] Ad Pipeline        : Video -> Canonical Ad Schema
  -> [Tier 2] Persona Generation  : Statistical distributions -> Twin persona pairs
  -> [Tier 3] Simulation Env      : Session context + ad insertion
  -> [Tier 4] LLM-Agent Simulation Loop : CoT-based baseline + causal response estimation
  -> [Tier 5] Statistical Analysis : Paired difference estimation + CI
  -> [Tier 6] Diagnosis            : Root cause attribution + improvements
  -> [Tier 7] Integration Outputs  : schemas + downstream report contracts
```

### Tier 1: Ad Pipeline (`src/ad_pipeline/`)

Processes raw video into a structured canonical schema:
- Video normalization (FFmpeg), shot segmentation (PySceneDetect)
- Audio transcription (ASR), on-screen text extraction (OCR)
- Visual feature tagging (emotion, brand/product visibility, price signals, CTA events)

### Tier 2: Persona Generation (`src/persona/`)

Generates personas as **state-variable collections** (not natural language profiles):
- Demographics from public statistics (ACS, CPS, CEX)
- Brand prior familiarity, ad tolerance, price sensitivity
- Interest vectors, search/click propensity, session behavior
- Twin-pair matching for controlled comparison (FR-09B)
- **Multi-affinity OR overlay:** supports single or multiple affinity presets via `build_affinity_templates_multi`; selected affinities are treated as disjoint pools and produce per-affinity `affinity_breakdown` metrics when ≥ 2 affinities are selected

### Tier 3-4: LLM-Agent Simulation Loop

The public core defines the schema contracts and protocol interfaces used by the LLM-Agent Simulation Loop:
1. **Call 0-A** (Segment Baseline): 7-metric baseline at the demographic cell level
2. **Call 0-B** (Individual Baseline): Per-persona adjustment anchored to segment means
3. **Treatment Call 1/2**: ad viewing narrative, then exposure-driven metric deltas
4. **Control Call 1/2**: no-ad session state, then near-zero counterfactual drift

Video input is first converted into the canonical ad schema. A deployment-specific runtime can pass that structured timeline into CoT-based AI-agent calls through the public protocol interfaces.

### Tier 5-6: Statistics + Diagnosis

- Paired difference estimation with confidence intervals
- Statistical power planning for required twin-pair counts and repeated seeds
- Root cause attribution and improvement hypothesis generation

### Tier 7: Integration Outputs

- JSON-compatible result contracts for downstream reporting
- Protocol interfaces for deployment-specific orchestration

---

## Repository Structure

This repository contains the public core modules exported for review and integration. Deployment-specific orchestration is intentionally kept outside the public core.

```
ai-video-ad-simulation-core/
├── src/
│   ├── config_loader.py          # TOML configuration loader
│   ├── ad_pipeline/              # Video processing pipeline (FFmpeg, ASR, OCR)
│   ├── ingestion/                # Public data schemas + normalization
│   │   ├── public_data_schemas.py
│   │   ├── ad_normalizer.py
│   │   ├── behavior_normalizer.py
│   │   ├── joint_demographic_distribution.py
│   │   ├── yt8m_vocabulary.py
│   │   └── ad_input_service.py
│   ├── persona/
│   │   ├── models.py             # Persona data models (demographics, priors)
│   │   └── targeting.py          # Audience targeting criteria
│   └── analysis/                 # Statistical power planning
├── core/
│   └── protocols.py              # Strategy-pattern interfaces (LLMClient, PromptProvider, SimulationOrchestrator)
├── config/
│   └── core.toml                 # Public configuration (power, pipeline defaults)
├── vocab/
│   └── youtube8m/vocabulary.csv  # Category vocabulary (kept outside data/ so the runtime volume mount does not hide it)
├── tests/                        # Unit tests for CORE modules
├── LICENSE.md                    # BSL 1.1
├── pyproject.toml
└── README.md
```

`core/protocols.py` still defines strategy-pattern interfaces for deployments that split proprietary orchestration into a separate package, but this working tree includes the current demo runtime implementation directly under `src/`.

---

## Key Design Principles

### Twin-World Separation
Complete isolation of treatment and control agents. Identical initial state, persona attributes, and external context. The **only** difference is ad exposure. Treatment agents receive the ad timeline in Call 1 and estimate exposure deltas in Call 2; control agents receive no ad timeline and estimate only no-ad baseline state plus small counterfactual drift. Paired design enables precise difference estimation.

### State-Variable Personas
Personas are numeric/categorical attribute vectors, not natural language profiles. This enables statistical sampling, reproducibility, and controlled variation within segments.

### Funnel Coherence Clamping
Metric outputs pass through deterministic post-hoc checks that preserve basic marketing funnel consistency across awareness, consideration, and intent. The public README describes the principle only; the exact constraint set is treated as implementation detail.

### Fallback-First Robustness
Deterministic rule-based fallbacks ensure the system produces results even when LLM calls fail or are unavailable.

---

## Repository Scope and Usage Requirements

This public core repository is **not a standalone runnable application**. Installing
`ai-video-ad-simulation-core` by itself does not provide the deployment-specific
orchestration needed to run an end-to-end ad simulation.

The public core is intended for architecture review, schema/model reuse, and
integration into a larger runtime package.

To use these modules in a working system, an integrating runtime must provide:

- Python >= 3.11
- FFmpeg installed and available on `PATH` when using the video ad pipeline
- Deployment-specific orchestration implementing the interfaces in `core/protocols.py`
- Hosted multimodal model access, prompt orchestration, credentials, and runtime
  configuration outside this public core
- Any proprietary calibration, reporting, web/API, or production execution layers
  required by the deployment

---

## Public Datasets

The system is designed to integrate with:
- **KuaiRand / KuaiRec**: Short video viewing behavior distributions
- **LAMBDA / UltraLAMBDA**: Ad memorability and creative features
- **YouTube-8M**: Video category vocabulary and content tags
- **ACS / CPS / CEX**: US demographic and expenditure statistics

---

## License

This project is licensed under the [Business Source License 1.1](LICENSE.md).

The BSL grants read access and non-production use immediately. After the Change Date (5 years from each release), the license converts to Apache 2.0.

**Additional Use Grant:** You may use the Licensed Work for evaluation, development, and testing purposes. Production use of the Licensed Work requires a separate commercial license.

---

## Citation

```bibtex
@software{ai_video_ad_simulation_2026,
  title={AI Video Ad Simulation: Synthetic Pretest System},
  year={2026},
  url={https://github.com/your-org/ai-video-ad-simulation-core}
}
```
