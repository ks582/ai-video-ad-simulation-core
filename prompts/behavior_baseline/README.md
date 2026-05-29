# LLM-Elicited Behavior Baseline Prompts

This directory hosts the **versioned prompt bundle** used by the monthly
LLM-elicited behavior baseline pipeline. Each `vN/` subdirectory contains
the exact files that get rendered, hashed, and shipped to elicitation models.

## Scope and license

This bundle is published as a **public assumption / reproducibility
artifact**, NOT as an empirical calibration. It documents the qualitative
elicitation methodology used to derive band-form behavioral priors so
that integrators can audit and reproduce the simulation's assumption
surface end-to-end.

Distributed under [BSL 1.1](../../LICENSE.md) (converts to Apache 2.0
after five years). License terms are declared at the repository root and
in the wheel `METADATA`; the prompt files themselves are kept verbatim
because every byte feeds the `prompt_hash` invariant (see
[Immutability rule](#immutability-rule-plan-r718--r817) below) — adding
a license header inside a hash-target file would change the snapshot
contract.

**Not included in this bundle** (intentionally private — see project
[Repository Scope](../../README.md#repository-scope-and-usage-requirements)):

- Empirical calibration data or campaign measurements
- LLM output post-processing / control logic
- Reporting, web/API, and production execution layers
- Proprietary orchestration that consumes these prompts at runtime

## Directory layout

```
prompts/behavior_baseline/
  README.md                            (this file)
  v1/
    system.txt                         (system prompt — fixed instructions)
    user.txt                           (user prompt template — append slot text)
    attribute_descriptions.json        (attribute → human-readable definition)
    output_schema.json                 (JSON Schema for per-model response)
  v2/                                  (future revision)
    ...
```

## Files inside each version directory

| File | Purpose | Hash-target? |
|------|---------|--------------|
| `system.txt` | Operating principles given to the model as the system role | yes |
| `user.txt` | Per-call task description. The renderer appends a generated `--- attributes ---` block (with `source_range_lo`/`source_range_hi` substituted from `DEFAULT_BOUNDS`) and the canonical JSON of `output_schema.json` after this text. | yes |
| `attribute_descriptions.json` | Top-level dict `{attribute_name: prose description}`. Used by the renderer to fill `attribute_description` slots. | yes (transitively, embedded in `rendered_user_text`) |
| `output_schema.json` | JSON Schema describing the expected per-model response shape. Embedded canonically (sorted keys, no whitespace) inside `rendered_user_text` so it participates in `prompt_hash`. Also used at generator time for post-hoc `jsonschema.validate` of LLM output. | yes (transitively) |

Every byte of these four files affects the recomputed `prompt_hash` checked
by invariant 10. Editing any of them retroactively invalidates every
existing snapshot that referenced this version.

## Immutability rule (plan R7.18 / R8.17)

**Versioned prompt directories are IMMUTABLE once a snapshot references them.**

Concretely:

- A snapshot stored under `behavior_baselines/YYYY-MM.json` with
  `"prompt_version": "v1"` pins itself to the byte content of
  `prompts/behavior_baseline/v1/` as it existed at generation time.
- The loader at `src.persona.behavior_priors.load_llm_elicited_snapshot()`
  re-renders the v1 bundle and recomputes `prompt_hash` every time it
  validates the snapshot. Any mutation of the v1 files (whitespace, JSON
  field order, additional commentary) changes the hash and causes the
  loader to fail-closed with an `LLMElicitedSnapshotError` whose message
  includes `[3/prompt_hash_mismatch]`. (`LLMElicitedSnapshotError` is a
  plain `ValueError` subclass with no category attribute; the violation
  category is encoded in the message text by the loader.)
- This fail-closed posture is intentional. Phase 1 has NO production
  fallback that disables prompt-hash verification. A wheel install that
  ships without these files fails to load any snapshot at all (plan H3
  Option A).

### How to update the prompt bundle

Do **not** modify `v1/` in place. Instead:

1. Create a new directory: `prompts/behavior_baseline/v2/`.
2. Copy v1's four files and edit them there.
3. The next monthly elicitation will use `--prompt-version v2` and write
   `behavior_baselines/_pending/YYYY-MM.json` with `"prompt_version": "v2"`.
4. Older snapshots continue to load against their original `v1/` bundle.

This append-only versioning preserves audit reproducibility for every
historical baseline.

## Deny-content rule (plan M5)

The export script (`scripts/export_core.sh`) runs a plain-string scan against
every exported file. Two private-side module references are banned from
appearing in any exported content (refer to the export script for the exact
patterns). No string literal, comment, or docstring in any file under this
directory — including this README — may contain those forbidden substrings.
Verify with the export script before editing prompt files.

## Distribution

These files are shipped in source-checkout AND in wheel installs (Hatch
`force-include` configured in `pyproject.toml`). The loader's invariant 10
relies on the bundle being co-located with the `src/persona/` modules
under the resolved `PROJECT_ROOT`.

## Phase 1 scope notice

The current `v1/` content is the initial Phase 1 elicitation prompt. The
prompt text deliberately avoids fabricated empirical statistics; it asks
models for qualitative, band-form priors only. These priors are
explicit simulation assumptions, NOT empirical calibrations.