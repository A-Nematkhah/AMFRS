"""
Canonical AMFRS end-to-end pipeline.

The former EvoNav Algorithm 1 orchestrator was removed; this module re-exports
the single AMFRS path from ``crowd_nav.reward_search.amfrs``.
"""

from crowd_nav.reward_search.amfrs import AMFRSArtifacts, AMFRSPipeline, AMFRSRunConfig

__all__ = [
    "AMFRSArtifacts",
    "AMFRSPipeline",
    "AMFRSRunConfig",
]
