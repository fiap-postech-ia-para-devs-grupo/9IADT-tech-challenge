from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AGExperiment:
    name: str
    population: int
    generations: int
    mutation_rate: float
    best_f1: float
    convergence: list[float]


@dataclass(frozen=True)
class BestConfig:
    experiment: str
    n_estimators: int
    max_depth: int
    min_samples_split: int
    max_features: str


@dataclass(frozen=True)
class AGResults:
    experiments: list[AGExperiment]
    baseline_f1: float
    best_config: BestConfig


def load_ag_results() -> AGResults:
    """Presentation-ready AG summary owned outside the HTTP adapter."""

    return AGResults(
        experiments=[
            AGExperiment(
                name="Exp 1",
                population=10,
                generations=30,
                mutation_rate=0.01,
                best_f1=0.951,
                convergence=[0.88, 0.91, 0.93, 0.94, 0.951],
            ),
            AGExperiment(
                name="Exp 2",
                population=30,
                generations=50,
                mutation_rate=0.01,
                best_f1=0.967,
                convergence=[0.89, 0.93, 0.95, 0.961, 0.967],
            ),
            AGExperiment(
                name="Exp 3",
                population=30,
                generations=50,
                mutation_rate=0.10,
                best_f1=0.943,
                convergence=[0.87, 0.90, 0.92, 0.935, 0.943],
            ),
        ],
        baseline_f1=0.934,
        best_config=BestConfig(
            experiment="Exp 2",
            n_estimators=300,
            max_depth=10,
            min_samples_split=2,
            max_features="sqrt",
        ),
    )
