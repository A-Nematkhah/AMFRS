"""
AMFRS package — multi-fidelity reward search (MAP-Elites, static gate, robustness).

Canonical entry: ``scripts/run_amfrs.py`` → ``AMFRSPipeline``.
"""

from crowd_nav.reward_search.amfrs.config import AMFRSRunConfig
from crowd_nav.reward_search.amfrs.pipeline import AMFRSArtifacts, AMFRSPipeline

__all__ = [
    "AMFRSArtifacts",
    "AMFRSPipeline",
    "AMFRSRunConfig",
]
