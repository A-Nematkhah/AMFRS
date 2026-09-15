"""
AMFRS v2 innovation package (additive; does not modify the EvoNav-faithful baseline).

See AMFRS_V2_INNOVATION_ROADMAP.md and the execution plan for axes 1–5.
"""

from crowd_nav.reward_search.amfrs.config import AMFRS2RunConfig
from crowd_nav.reward_search.amfrs.pipeline2 import AMFRS2Artifacts, AMFRS2Pipeline

__all__ = [
    "AMFRS2Artifacts",
    "AMFRS2Pipeline",
    "AMFRS2RunConfig",
]
