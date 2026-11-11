from .autoregressive import AutoregressiveGenerator
from .base import Generator
from .composite import CompositeGenerator
from .events_counts import EventsCountsGenerator
from .gaussian_process import GaussianProcessGenerator
from .periodic import ComplexPeriodicGenerator
from .regime_sde import RegimeSDEGenerator
from .trend_seasonal import TrendSeasonalGenerator

# The seven families of the reference prior (G1..G7).
REGISTRY: dict[str, type[Generator]] = {
    TrendSeasonalGenerator.family: TrendSeasonalGenerator,  # G1
    ComplexPeriodicGenerator.family: ComplexPeriodicGenerator,  # G2
    GaussianProcessGenerator.family: GaussianProcessGenerator,  # G3
    AutoregressiveGenerator.family: AutoregressiveGenerator,  # G4
    RegimeSDEGenerator.family: RegimeSDEGenerator,  # G5
    EventsCountsGenerator.family: EventsCountsGenerator,  # G6
    CompositeGenerator.family: CompositeGenerator,  # G7 (built from the others)
}

__all__ = [
    "REGISTRY",
    "Generator",
    "TrendSeasonalGenerator",
    "ComplexPeriodicGenerator",
    "GaussianProcessGenerator",
    "AutoregressiveGenerator",
    "RegimeSDEGenerator",
    "EventsCountsGenerator",
    "CompositeGenerator",
]
