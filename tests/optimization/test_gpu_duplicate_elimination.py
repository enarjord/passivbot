import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.population import Population

from optimization.backends.gpu_backend import NativeSearchProblem
from optimization.backends.pymoo_backend import _build_algorithm, _resolve_pymoo_population_plan
from optimization.bounds import Bound
from optimization.gpu import duplicate_elimination
from optimization.gpu.duplicate_elimination import BoundedDuplicateElimination


@pytest.mark.parametrize("budget", [8 * 5, 8 * 12 * 2])
@pytest.mark.parametrize("to_itself", [False, True])
def test_blocked_distances_preserve_pymoo_order_and_tolerance(monkeypatch, to_itself, budget):
    candidates = Population.new("X", np.asarray([
        [0., 0.], [1., 1.], [1., 1.], [1., 2.],
        [np.nan, 2.], [0.8e-16, 0.], [1.6e-16, 0.], [np.nan, 2.],
        [np.inf, 2.], [np.inf, 2.], [3., 3.], [4., 4.],
    ]))
    parents = Population.new("X", np.asarray([[1., 2.], [6., 6.]]))
    offspring = Population.new("X", np.asarray([[3., 3.]]))
    monkeypatch.setattr(BoundedDuplicateElimination, "_DISTANCE_SCRATCH_BYTES", budget)
    observed_blocks = []
    full_cdist = duplicate_elimination.cdist

    def measure(left, right):
        result = full_cdist(left, right)
        assert result.nbytes <= budget
        observed_blocks.append(result.shape)
        return result

    monkeypatch.setattr(duplicate_elimination, "cdist", measure)
    policy = BoundedDuplicateElimination()
    assert policy.epsilon == DefaultDuplicateElimination().epsilon
    for against in ((), (Population.create(),), (parents, offspring)):
        expected = DefaultDuplicateElimination().do(candidates, *against, return_indices=True, to_itself=to_itself)
        actual = policy.do(candidates, *against, return_indices=True, to_itself=to_itself)
        np.testing.assert_array_equal(actual[0].get("X"), expected[0].get("X"))
        assert actual[1:] == expected[1:]
    assert observed_blocks
    if to_itself:
        expected_shape = (1, 5) if budget == 8 * 5 else (2, len(candidates))
        assert expected_shape in observed_blocks


@pytest.mark.parametrize("algorithm_name,n_obj", [("nsga2", 2), ("nsga3", 4)])
def test_max_operator_preserves_quantized_ask_after_serializing_algorithm(monkeypatch, algorithm_name, n_obj):
    bounds = [Bound(0, 1, 0.05), Bound(0, 1, 0.05), Bound(0, 1), Bound(.3, .3)]
    problem = NativeSearchProblem(SimpleNamespace(bounds=bounds, scoring_specs=[None] * n_obj))
    config = {"optimize": {"population_size": 8, "pymoo": {"algorithm": algorithm_name}}}
    plan = _resolve_pymoo_population_plan(config, n_obj=n_obj)
    vector = np.linspace(0, 1, 8)
    sampling = np.column_stack((vector, vector[::-1], vector, np.full(8, .3)))
    monkeypatch.setattr(BoundedDuplicateElimination, "_DISTANCE_SCRATCH_BYTES", 8 * 128 * 2)

    def create(bounded):
        algorithm = _build_algorithm(config=config, sampling=sampling, bounds=bounds, sig_digits=4,
                                     population_plan=plan, bounded_duplicates=bounded)
        algorithm.setup(problem, seed=123, verbose=False)
        parents = algorithm.ask()
        genes = parents.get("X")
        parents.set("F", np.asarray([genes[:, i % len(bounds)] for i in range(n_obj)]).T,
                    "G", -np.ones((len(genes), 1)))
        for individual in parents:
            individual.evaluated.update(("F", "G", "H"))
        algorithm.tell(infills=parents)
        algorithm.n_offsprings = 128
        return algorithm

    canonical = create(False)
    bounded = pickle.loads(pickle.dumps(create(True)))
    assert type(canonical.eliminate_duplicates) is DefaultDuplicateElimination
    assert type(bounded.eliminate_duplicates) is BoundedDuplicateElimination
    assert bounded.initialization.eliminate_duplicates is bounded.mating.eliminate_duplicates is bounded.eliminate_duplicates
    old, new = canonical.ask(), bounded.ask()
    assert len(new) == 128
    np.testing.assert_array_equal(new.get("X"), old.get("X"))
    assert bounded.random_state.bit_generator.state == canonical.random_state.bit_generator.state

    config["optimize"]["pymoo"]["shared"] = {"eliminate_duplicates": False}
    disabled = create(True)
    assert type(disabled.eliminate_duplicates) is NoDuplicateElimination
