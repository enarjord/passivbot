import os
import json
import hashlib
from typing import Any, Dict
import glob
import math
from opt_utils import calc_normalized_dist


def round_floats(obj: Any, sig_digits: int = 6) -> Any:
    if isinstance(obj, float):
        return float(f"{obj:.{sig_digits}g}")
    elif isinstance(obj, dict):
        return {k: round_floats(v, sig_digits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v, sig_digits) for v in obj]
    else:
        return obj


def hash_entry(entry: Dict) -> str:
    entry_str = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(entry_str.encode("utf-8")).hexdigest()[:16]


def calc_dist(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


class ParetoStore:
    def __init__(self, directory: str, sig_digits: int = 6):
        self.directory = directory
        self.sig_digits = sig_digits
        os.makedirs(os.path.join(self.directory, "pareto"), exist_ok=True)
        self.index_path = os.path.join(self.directory, "index.json")
        self.hashes = set()
        self._load_index()

    def hash_entry(self, entry):
        return hash_entry(entry)

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    self.hashes = set(json.load(f))
            except Exception as e:
                print(f"Failed to load Pareto index: {e}")

    def _save_index(self):
        tmp_path = self.index_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(sorted(self.hashes), f)
        os.replace(tmp_path, self.index_path)

    def add_entry(self, entry: Dict) -> bool:
        rounded = round_floats(entry, self.sig_digits)
        h = hash_entry(rounded)
        if h in self.hashes:
            return False  # already exists

        # Save entry without distance prefix
        filename = f"{h}.json"
        filepath = os.path.join(self.directory, "pareto", filename)
        tmp_path = filepath + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(rounded, f, separators=(",", ":"), indent=4, sort_keys=True)
            os.replace(tmp_path, filepath)
            self.hashes.add(h)
            self._save_index()

            # Recompute distances and rename all
            self.rename_entries_with_distance()
            return True
        except Exception as e:
            print(f"Failed to write Pareto entry {h}: {e}")
            return False

    def remove_entry(self, hash_str: str) -> bool:
        pattern = os.path.join(self.directory, "pareto", f"*_{hash_str}.json")
        matches = glob.glob(pattern)
        success = False
        for filepath in matches:
            try:
                os.remove(filepath)
                success = True
            except Exception as e:
                print(f"Failed to remove file {filepath}: {e}")
        if hash_str in self.hashes:
            self.hashes.remove(hash_str)
            self._save_index()
        return success

    def list_entries(self) -> list:
        return sorted(self.hashes)

    def load_entry(self, hash_str: str) -> Dict:
        pattern = os.path.join(self.directory, "pareto", f"*_{hash_str}.json")
        matches = glob.glob(pattern)
        if not matches:
            pattern_alt = os.path.join(self.directory, "pareto", f"{hash_str}.json")
            matches = glob.glob(pattern_alt)
        if not matches:
            raise FileNotFoundError(f"No entry found for hash {hash_str}")
        with open(matches[0], "r") as f:
            return json.load(f)

    def clear(self):
        for h in list(self.hashes):
            self.remove_entry(h)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        self.hashes.clear()

    def compute_ideal_point(self):
        entries = [self.load_entry(h) for h in self.hashes]
        w_keys = [k for k in entries[0].get("analyses_combined", {}) if k.startswith("w_")]
        if not entries or not w_keys:
            return None
        ideal = []
        for key in sorted(w_keys):
            vals = [
                e["analyses_combined"].get(key)
                for e in entries
                if e["analyses_combined"].get(key) is not None
            ]
            if not vals:
                return None
            ideal.append(min(vals))
        return tuple(ideal)

    def rename_entries_with_distance(self):
        ideal = self.compute_ideal_point()
        if ideal is None:
            print("No valid entries to compute ideal point.")
            return

        # Get all keys w_0, w_1, ..., w_n
        sample_entry = self.load_entry(next(iter(self.hashes)))
        w_keys = sorted(k for k in sample_entry.get("analyses_combined", {}) if k.startswith("w_"))
        if not w_keys:
            print("No w_i keys found.")
            return

        dims = len(w_keys)
        value_matrix = [[] for _ in range(dims)]
        for h in self.hashes:
            entry = self.load_entry(h)
            for i, key in enumerate(w_keys):
                val = entry.get("analyses_combined", {}).get(key)
                if val is not None:
                    value_matrix[i].append(val)

        mins = [min(vals) for vals in value_matrix]
        maxs = [max(vals) for vals in value_matrix]

        for h in sorted(self.hashes):
            try:
                entry = self.load_entry(h)
                point = []
                for i, key in enumerate(w_keys):
                    val = entry.get("analyses_combined", {}).get(key)
                    if val is None:
                        val = float("inf")
                    denom = maxs[i] - mins[i]
                    norm = (val - mins[i]) / denom if denom > 0 else 0.0
                    point.append(norm)
                ideal_norm = [
                    (ideal[i] - mins[i]) / (maxs[i] - mins[i]) if (maxs[i] - mins[i]) > 0 else 0.0
                    for i in range(dims)
                ]
                dist = math.sqrt(sum((p - i) ** 2 for p, i in zip(point, ideal_norm)))
                dist_prefix = f"{dist:08.4f}"

                old_path_pattern = os.path.join(self.directory, "pareto", f"*{h}.json")
                old_matches = glob.glob(old_path_pattern)
                if not old_matches:
                    continue
                old_path = old_matches[0]
                new_path = os.path.join(self.directory, "pareto", f"{dist_prefix}_{h}.json")
                os.rename(old_path, new_path)
            except Exception as e:
                print(f"Failed to rename entry {h}: {e}")


def main():
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    import json

    parser = argparse.ArgumentParser(description="Analyze and plot Pareto front")
    parser.add_argument("pareto_dir", type=str, help="Path to pareto/ directory")
    parser.add_argument("--json", action="store_true", help="Output summary as JSON")
    args = parser.parse_args()

    store = ParetoStore(os.path.dirname(args.pareto_dir.rstrip("/")))
    print(f"Found {len(store.hashes)} Pareto members.")

    points = []
    for h in store.hashes:
        try:
            entry = store.load_entry(h)
            w0 = entry.get("analyses_combined", {}).get("w_0")
            w1 = entry.get("analyses_combined", {}).get("w_1")
            if w0 is not None and w1 is not None:
                points.append((w0, w1, h))
        except Exception as e:
            print(f"Error loading {h}: {e}")

    if not points:
        print("No valid Pareto points found.")
        exit(0)

    w0s, w1s, hashes = zip(*points)
    ideal = (min(w0s), min(w1s))

    # Normalized distance calculation
    w0_arr = np.array(w0s)
    w1_arr = np.array(w1s)
    w0_min, w0_max = min(w0_arr), max(w0_arr)
    w1_min, w1_max = min(w1_arr), max(w1_arr)
    norm_w0 = (w0_arr - w0_min) / (w0_max - w0_min) if w0_max > w0_min else w0_arr
    norm_w1 = (w1_arr - w1_min) / (w1_max - w1_min) if w1_max > w1_min else w1_arr
    ideal_norm = ((ideal[0] - w0_min) / (w0_max - w0_min), (ideal[1] - w1_min) / (w1_max - w1_min))
    dists = np.sqrt((norm_w0 - ideal_norm[0]) ** 2 + (norm_w1 - ideal_norm[1]) ** 2)
    closest_idx = int(np.argmin(dists))

    print(f"Ideal point: w_0={ideal[0]:.5f}, w_1={ideal[1]:.5f}")
    print(f"Closest to ideal: {hashes[closest_idx]} | norm_dist={dists[closest_idx]:.5f}")
    print(f"w_0={w0s[closest_idx]:.5f}, w_1={w1s[closest_idx]:.5f}")

    if args.json:
        summary = {
            "n_members": len(hashes),
            "ideal": {"w_0": ideal[0], "w_1": ideal[1]},
            "closest": {
                "hash": hashes[closest_idx],
                "w_0": w0s[closest_idx],
                "w_1": w1s[closest_idx],
                "normalized_distance": dists[closest_idx],
            },
        }
        print(json.dumps(summary, indent=4))

    plt.scatter(w0s, w1s, label="Pareto Members")
    plt.scatter(*ideal, color="green", label="Ideal Point", zorder=5)
    plt.scatter(w0s[closest_idx], w1s[closest_idx], color="red", label="Closest to Ideal", zorder=5)
    plt.xlabel("w_0")
    plt.ylabel("w_1")
    plt.title("Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
