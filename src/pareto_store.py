import os
import json
import hashlib
from typing import Dict
import glob
import math
from opt_utils import calc_normalized_dist, round_floats


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

    pareto_dir = args.pareto_dir.rstrip("/")
    entries = sorted(glob.glob(os.path.join(pareto_dir, "*.json")))
    print(f"Found {len(entries)} Pareto members.")

    store = ParetoStore(os.path.dirname(args.pareto_dir.rstrip("/")))
    print(f"Found {len(store.hashes)} Pareto members.")

    points = []
    w_keys = []
    for entry_path in entries:
        try:
            with open(entry_path) as f:
                entry = json.load(f)
            h = os.path.splitext(os.path.basename(entry_path))[0].split("_")[-1]
            if not w_keys:
                w_keys = sorted(k for k in entry.get("analyses_combined", {}) if k.startswith("w_"))
            values = [entry.get("analyses_combined", {}).get(k) for k in w_keys]
            if all(v is not None for v in values):
                points.append((*values, h))
        except Exception as e:
            print(f"Error loading {h}: {e}")

    if not points:
        print("No valid Pareto points found.")
        exit(0)

    values_matrix = np.array([p[:-1] for p in points])
    hashes = [p[-1] for p in points]

    ideal = np.min(values_matrix, axis=0)
    mins = np.min(values_matrix, axis=0)
    maxs = np.max(values_matrix, axis=0)

    norm_matrix = np.array(
        [
            [
                (v - mins[i]) / (maxs[i] - mins[i]) if maxs[i] > mins[i] else v
                for i, v in enumerate(row)
            ]
            for row in values_matrix
        ]
    )
    ideal_norm = [
        (ideal[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] > mins[i] else ideal[i]
        for i in range(len(ideal))
    ]

    dists = np.linalg.norm(norm_matrix - ideal_norm, axis=1)
    closest_idx = int(np.argmin(dists))

    print("Ideal point:")
    for i, key in enumerate(w_keys):
        print(f"  {key} = {ideal[i]:.5f}")
    print(f"Closest to ideal: {hashes[closest_idx]} | norm_dist={dists[closest_idx]:.5f}")
    for i, key in enumerate(w_keys):
        print(f"  {key} = {values_matrix[closest_idx][i]:.5f}")

    if args.json:
        summary = {
            "n_members": len(hashes),
            "ideal": {k: float(ideal[i]) for i, k in enumerate(w_keys)},
            "closest": {
                "hash": hashes[closest_idx],
                **{k: float(values_matrix[closest_idx][i]) for i, k in enumerate(w_keys)},
                "normalized_distance": float(dists[closest_idx]),
            },
        }
        print(json.dumps(summary, indent=4))

    fig = plt.figure(figsize=(12, 4))

    if len(w_keys) == 2:
        ax = fig.add_subplot(111)
        ax.scatter(values_matrix[:, 0], values_matrix[:, 1], label="Pareto Members")
        ax.scatter(*ideal, color="green", label="Ideal Point", zorder=5)
        ax.scatter(
            values_matrix[closest_idx][0],
            values_matrix[closest_idx][1],
            color="red",
            label="Closest to Ideal",
            zorder=5,
        )
        ax.set_xlabel(w_keys[0])
        ax.set_ylabel(w_keys[1])
        ax.set_title("Pareto Front")
        ax.legend()
        ax.grid(True)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    elif len(w_keys) == 3:
        import plotly.graph_objs as go
        import plotly.io as pio

        # Load a config to get metric names
        sample_entry_path = entries[0]
        with open(sample_entry_path) as f:
            sample_entry = json.load(f)

        # Try to read optimize.scoring if available
        metric_names = sample_entry.get("optimize", {}).get("scoring", [])
        metric_name_map = {f"w_{i}": name for i, name in enumerate(metric_names)}

        fig = go.Figure()

        # Scatter points for Pareto members
        fig.add_trace(
            go.Scatter3d(
                x=values_matrix[:, 0],
                y=values_matrix[:, 1],
                z=values_matrix[:, 2],
                mode="markers",
                marker=dict(size=4, color="blue"),
                name="Pareto Members",
                text=[f"hash: {h}" for h in hashes],
                hoverinfo="text",
            )
        )

        # Ideal point
        fig.add_trace(
            go.Scatter3d(
                x=[ideal[0]],
                y=[ideal[1]],
                z=[ideal[2]],
                mode="markers",
                marker=dict(size=8, color="green"),
                name="Ideal Point",
            )
        )

        # Closest point
        fig.add_trace(
            go.Scatter3d(
                x=[values_matrix[closest_idx][0]],
                y=[values_matrix[closest_idx][1]],
                z=[values_matrix[closest_idx][2]],
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Closest to Ideal",
            )
        )

        fig.update_layout(
            title="Pareto Front (3D Interactive)",
            scene=dict(
                xaxis_title=metric_name_map.get(w_keys[0], w_keys[0]),
                yaxis_title=metric_name_map.get(w_keys[1], w_keys[1]),
                zaxis_title=metric_name_map.get(w_keys[2], w_keys[2]),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(x=0.01, y=0.99),
        )

        fig.show()
    else:
        ax = fig.add_subplot(111)
        ax.plot(range(len(w_keys)), values_matrix[closest_idx], marker="o")
        ax.set_xticks(range(len(w_keys)))
        ax.set_xticklabels(w_keys, rotation=45)
        ax.set_title("Closest to Ideal Point")
        ax.grid(True)

        ax = fig.add_subplot(111, projection="3d" if len(w_keys) == 3 else None)
        if len(w_keys) == 3:
            ax.scatter(
                values_matrix[:, 0], values_matrix[:, 1], values_matrix[:, 2], label="Pareto Members"
            )
            ax.set_xlabel(w_keys[0])
            ax.set_ylabel(w_keys[1])
            ax.set_zlabel(w_keys[2])
        else:
            ax.plot(range(len(w_keys)), values_matrix[closest_idx], marker="o")
            ax.set_xticks(range(len(w_keys)))
            ax.set_xticklabels(w_keys, rotation=45)
            ax.set_title("Closest to Ideal Point")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
