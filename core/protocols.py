"""Strategy-pattern interfaces for the SaaS layer.

The public core repository defines these Protocol interfaces.
The private SaaS repository provides concrete implementations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Abstraction over LLM API providers (Gemini, OpenAI, Anthropic)."""

    def call(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...


@runtime_checkable
class PromptProvider(Protocol):
    """Builds LLM prompts for each simulation phase."""

    def build_call0_prompt(self, persona: Any, ad_schema: Any) -> str: ...

    def build_call1_prompt(self, persona: Any, ad_schema: Any) -> str: ...

    def build_call2_prompt(self, persona: Any, ad_schema: Any) -> str: ...


@runtime_checkable
class SimulationOrchestrator(Protocol):
    """Coordinates the multi-phase simulation pipeline."""

    def run(self, *, config: dict[str, Any]) -> Any: ...