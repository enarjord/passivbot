import os
import json
import hashlib
from typing import Any, Dict
import glob
import math


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
        w0s = []
        w1s = []
        for h in self.hashes:
            try:
                entry = self.load_entry(h)
                w0 = entry.get("analyses_combined", {}).get("w_0")
                w1 = entry.get("analyses_combined", {}).get("w_1")
                if w0 is not None and w1 is not None:
                    w0s.append(w0)
                    w1s.append(w1)
            except:
                continue
        if w0s and w1s:
            return (min(w0s), min(w1s))
        return None

    def rename_entries_with_distance(self):
        ideal = self.compute_ideal_point()
        if ideal is None:
            print("No valid entries to compute ideal point.")
            return
        for h in sorted(self.hashes):
            try:
                entry = self.load_entry(h)
                w0 = entry.get("analyses_combined", {}).get("w_0")
                w1 = entry.get("analyses_combined", {}).get("w_1")
                if w0 is None or w1 is None:
                    dist_prefix = "9999.9999"
                else:
                    dist = calc_dist((w0, w1), ideal)
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
