"""Planner registry for the supported SIMS generation task."""

from typing import Dict, Type

from sims.data_generation.path_planners import (
    HouseWalkthroughPlanner,
    PathPlanner,
)

REGISTERED_PLANNERS: Dict[str, Type[PathPlanner]] = {
    "HouseWalkthrough": HouseWalkthroughPlanner,
}

__all__ = ["REGISTERED_PLANNERS"]
