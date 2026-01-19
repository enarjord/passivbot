"""Pydantic models for Optuna optimizer configuration."""
from __future__ import annotations
from typing import Annotated, Literal, Self, Union
from pydantic import BaseModel, Field, model_validator


class Objective(BaseModel):
    """An optimization objective with explicit direction."""
    metric: str
    direction: Literal["maximize", "minimize"]

    @property
    def sign(self) -> float:
        """Return -1 for maximize (negate to minimize), 1 for minimize."""
        return -1.0 if self.direction == "maximize" else 1.0


class Constraint(BaseModel):
    """A metric constraint. At least one of min or max required."""
    metric: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def check_min_or_max(self) -> Constraint:
        if self.min is None and self.max is None:
            raise ValueError("At least one of min or max must be specified")
        return self


class Bound(BaseModel):
    """Parameter bound: continuous [low, high], stepped [low, high, step], or fixed."""
    low: float
    high: float
    step: float | None = None

    @classmethod
    def from_config(cls, value: float | list) -> Self:
        if isinstance(value, (int, float)):
            return cls(low=float(value), high=float(value))
        if len(value) == 1:
            return cls(low=float(value[0]), high=float(value[0]))
        if len(value) == 2:
            return cls(low=float(value[0]), high=float(value[1]))
        if len(value) == 3:
            return cls(low=float(value[0]), high=float(value[1]), step=float(value[2]))
        raise ValueError(f"Invalid bound format: {value}")

    @property
    def is_fixed(self) -> bool:
        return self.low == self.high

    @property
    def is_stepped(self) -> bool:
        return self.step is not None and self.step > 0


class NSGAIISamplerConfig(BaseModel):
    model_config = {"extra": "ignore"}

    name: Literal["nsgaii"] = "nsgaii"
    population_size: int = 250
    mutation_prob: float | None = None
    crossover_prob: float = 0.9
    seed: int | None = None


class NSGAIIISamplerConfig(BaseModel):
    model_config = {"extra": "ignore"}

    name: Literal["nsgaiii"] = "nsgaiii"
    population_size: int = 250
    mutation_prob: float | None = None
    crossover_prob: float = 0.9
    seed: int | None = None


SamplerConfig = Annotated[
    Union[NSGAIISamplerConfig, NSGAIIISamplerConfig],
    Field(discriminator="name"),
]


class OptunaConfig(BaseModel):
    """Main configuration for the Optuna optimizer."""
    n_trials: int = 250000
    n_cpus: int = 8
    max_best_trials: int = 200
    study_name: str | None = None
    sampler: SamplerConfig = Field(default_factory=NSGAIISamplerConfig)
