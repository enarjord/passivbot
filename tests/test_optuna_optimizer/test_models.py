import pytest
from pydantic import ValidationError


class TestObjective:
    def test_maximize(self):
        from optuna_optimizer.models import Objective
        obj = Objective(metric="adg_pnl", direction="maximize")
        assert obj.metric == "adg_pnl"
        assert obj.direction == "maximize"
        assert obj.sign == -1.0

    def test_minimize(self):
        from optuna_optimizer.models import Objective
        obj = Objective(metric="drawdown_worst", direction="minimize")
        assert obj.metric == "drawdown_worst"
        assert obj.direction == "minimize"
        assert obj.sign == 1.0

    def test_invalid_direction(self):
        from optuna_optimizer.models import Objective
        with pytest.raises(ValidationError):
            Objective(metric="adg_pnl", direction="invalid")


class TestConstraint:
    def test_with_max(self):
        from optuna_optimizer.models import Constraint
        c = Constraint(metric="drawdown_worst", max=0.5)
        assert c.metric == "drawdown_worst"
        assert c.max == 0.5
        assert c.min is None

    def test_with_min(self):
        from optuna_optimizer.models import Constraint
        c = Constraint(metric="adg_pnl", min=0.001)
        assert c.min == 0.001
        assert c.max is None

    def test_with_both(self):
        from optuna_optimizer.models import Constraint
        c = Constraint(metric="sharpe_ratio", min=0.5, max=3.0)
        assert c.min == 0.5
        assert c.max == 3.0

    def test_requires_min_or_max(self):
        from optuna_optimizer.models import Constraint
        with pytest.raises(ValidationError, match="At least one of min or max"):
            Constraint(metric="drawdown_worst")


class TestBound:
    def test_continuous(self):
        from optuna_optimizer.models import Bound
        b = Bound.from_config([0.001, 0.025])
        assert b.low == 0.001
        assert b.high == 0.025
        assert b.step is None
        assert not b.is_fixed
        assert not b.is_stepped

    def test_stepped(self):
        from optuna_optimizer.models import Bound
        b = Bound.from_config([200, 1440, 10])
        assert b.low == 200
        assert b.high == 1440
        assert b.step == 10
        assert b.is_stepped

    def test_fixed_range(self):
        from optuna_optimizer.models import Bound
        b = Bound.from_config([0.5, 0.5])
        assert b.is_fixed

    def test_fixed_single(self):
        from optuna_optimizer.models import Bound
        b = Bound.from_config(0.5)
        assert b.is_fixed
        assert b.low == 0.5


class TestSamplerConfig:
    def test_nsgaii(self):
        from optuna_optimizer.models import NSGAIISamplerConfig
        cfg = NSGAIISamplerConfig(population_size=100)
        assert cfg.name == "nsgaii"
        assert cfg.population_size == 100

    def test_nsgaiii(self):
        from optuna_optimizer.models import NSGAIIISamplerConfig
        cfg = NSGAIIISamplerConfig()
        assert cfg.name == "nsgaiii"

    def test_discriminated_union(self):
        from optuna_optimizer.models import SamplerConfig
        from pydantic import TypeAdapter

        adapter = TypeAdapter(SamplerConfig)

        cfg = adapter.validate_python({"name": "nsgaii", "population_size": 75})
        assert cfg.population_size == 75

        cfg = adapter.validate_python({"name": "nsgaiii", "population_size": 100})
        assert cfg.name == "nsgaiii"


class TestOptunaConfig:
    def test_defaults(self):
        from optuna_optimizer.models import OptunaConfig
        cfg = OptunaConfig()
        assert cfg.n_trials == 250000
        assert cfg.n_cpus == 8
        assert cfg.sampler.name == "nsgaii"

    def test_custom(self):
        from optuna_optimizer.models import OptunaConfig
        cfg = OptunaConfig(
            n_trials=5000,
            sampler={"name": "nsgaii", "population_size": 100},
        )
        assert cfg.n_trials == 5000
        assert cfg.sampler.name == "nsgaii"
