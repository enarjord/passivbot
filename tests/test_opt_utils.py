import msgpack

from opt_utils import apply_diffs, generate_diffs, generate_incremental_diff, load_results


def test_incremental_diff_replays_top_level_key_deletions():
    previous = {"config": {"a": 1}, "metrics": {"error": "bad"}}
    current = {"config": {"a": 2}}

    diff = generate_incremental_diff(previous, current)
    replayed = list(apply_diffs([diff], base=previous))[-1]

    assert "metrics" not in replayed
    assert replayed == current


def test_incremental_diff_replays_nested_key_deletions():
    previous = {
        "metrics": {
            "objectives": {"adg": 1.0, "mdg": 2.0},
            "error": "bad",
        }
    }
    current = {"metrics": {"objectives": {"adg": 1.5}}}

    diff = generate_incremental_diff(previous, current)
    replayed = list(apply_diffs([diff], base=previous))[-1]

    assert replayed == current


def test_load_results_replays_deleted_keys_from_incremental_diffs(tmp_path):
    first = {"config": {"a": 1}, "metrics": {"error": "bad", "gain": 0.0}}
    second = {"config": {"a": 2}, "metrics": {"gain": 1.0}}
    third = {"config": {"a": 3}}
    second_diff = generate_incremental_diff(first, second)
    third_diff = generate_incremental_diff(second, third)
    path = tmp_path / "all_results.bin"
    packer = msgpack.Packer(use_bin_type=True)
    with open(path, "wb") as f:
        f.write(packer.pack(first))
        f.write(packer.pack(second_diff))
        f.write(packer.pack(third_diff))

    assert list(load_results(path)) == [first, second, third]


def test_generate_diffs_replays_deleted_keys():
    first = {"a": 1, "stale": {"nested": True}}
    second = {"a": 2, "stale": {}}
    third = {"a": 3}

    diffs = list(generate_diffs([first, second, third]))

    assert list(apply_diffs(diffs)) == [first, second, third]
