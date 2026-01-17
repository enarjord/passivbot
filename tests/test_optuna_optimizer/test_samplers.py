from unittest.mock import MagicMock

import optuna
import pytest

from optuna_optimizer.models import (
    GPSamplerConfig,
    NSGAIISamplerConfig,
    NSGAIIISamplerConfig,
    RandomSamplerConfig,
    TPESamplerConfig,
)
from optuna_optimizer.samplers import get_sampler_config_by_name


class TestMakeConstraintsFunc:
    def test_returns_violations_from_user_attrs(self):
        from optuna_optimizer.samplers import make_constraints_func

        constraints_func = make_constraints_func()

        trial = MagicMock()
        trial.user_attrs = {"constraint_violations": [0.1, 0.0, 0.05]}

        result = constraints_func(trial)

        assert result == [0.1, 0.0, 0.05]

    def test_returns_empty_list_when_no_violations(self):
        from optuna_optimizer.samplers import make_constraints_func

        constraints_func = make_constraints_func()

        trial = MagicMock()
        trial.user_attrs = {}

        result = constraints_func(trial)

        assert result == []


class TestGetSamplerConfigByName:
    def test_tpe_returns_tpe_config(self):
        config = get_sampler_config_by_name("tpe")
        assert isinstance(config, TPESamplerConfig)

    def test_nsgaii_returns_nsgaii_config(self):
        config = get_sampler_config_by_name("nsgaii")
        assert isinstance(config, NSGAIISamplerConfig)

    def test_nsgaiii_returns_nsgaiii_config(self):
        config = get_sampler_config_by_name("nsgaiii")
        assert isinstance(config, NSGAIIISamplerConfig)

    def test_gp_returns_gp_config(self):
        config = get_sampler_config_by_name("gp")
        assert isinstance(config, GPSamplerConfig)

    def test_random_returns_random_config(self):
        config = get_sampler_config_by_name("random")
        assert isinstance(config, RandomSamplerConfig)

    def test_unknown_name_raises_key_error(self):
        with pytest.raises(KeyError):
            get_sampler_config_by_name("unknown")


class TestCreateSampler:
    def test_tpe(self):
        from optuna_optimizer.models import TPESamplerConfig
        from optuna_optimizer.samplers import create_sampler
        sampler = create_sampler(TPESamplerConfig())
        assert isinstance(sampler, optuna.samplers.TPESampler)

    def test_nsgaii(self):
        from optuna_optimizer.models import NSGAIISamplerConfig
        from optuna_optimizer.samplers import create_sampler
        sampler = create_sampler(NSGAIISamplerConfig(population_size=100))
        assert isinstance(sampler, optuna.samplers.NSGAIISampler)

    def test_nsgaiii(self):
        from optuna_optimizer.models import NSGAIIISamplerConfig
        from optuna_optimizer.samplers import create_sampler
        sampler = create_sampler(NSGAIIISamplerConfig())
        assert isinstance(sampler, optuna.samplers.NSGAIIISampler)

    def test_gp(self):
        from optuna_optimizer.models import GPSamplerConfig
        from optuna_optimizer.samplers import create_sampler
        sampler = create_sampler(GPSamplerConfig())
        assert isinstance(sampler, optuna.samplers.GPSampler)

    def test_random(self):
        from optuna_optimizer.models import RandomSamplerConfig
        from optuna_optimizer.samplers import create_sampler
        sampler = create_sampler(RandomSamplerConfig(seed=42))
        assert isinstance(sampler, optuna.samplers.RandomSampler)


class TestCreateSamplerWithConstraints:
    def test_nsgaii_with_constraints_func(self):
        from optuna_optimizer.models import NSGAIISamplerConfig
        from optuna_optimizer.samplers import create_sampler

        func = lambda t: [0.0]
        sampler = create_sampler(NSGAIISamplerConfig(), constraints_func=func)

        assert isinstance(sampler, optuna.samplers.NSGAIISampler)
        assert sampler._constraints_func is func

    def test_nsgaii_without_constraints_func(self):
        from optuna_optimizer.models import NSGAIISamplerConfig
        from optuna_optimizer.samplers import create_sampler

        sampler = create_sampler(NSGAIISamplerConfig())

        assert sampler._constraints_func is None

    def test_tpe_with_constraints_func(self):
        from optuna_optimizer.models import TPESamplerConfig
        from optuna_optimizer.samplers import create_sampler

        func = lambda t: [0.0]
        sampler = create_sampler(TPESamplerConfig(), constraints_func=func)

        assert isinstance(sampler, optuna.samplers.TPESampler)
        assert sampler._constraints_func is func

    def test_gp_with_constraints_func(self):
        from optuna_optimizer.models import GPSamplerConfig
        from optuna_optimizer.samplers import create_sampler

        func = lambda t: [0.0]
        sampler = create_sampler(GPSamplerConfig(), constraints_func=func)

        assert isinstance(sampler, optuna.samplers.GPSampler)
        assert sampler._constraints_func is func
