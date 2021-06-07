from __future__ import annotations
from typing import Any, Dict, Iterator
from nptyping import NDArray, Float64
from operator import mul
from operator import truediv as div
from typing import Optional
from copy import deepcopy
from pandas.io.json._normalize import nested_to_record
from njit_funcs import backtest
from analyze import analyze_backtest
from pure_funcs import ts_to_date
import numpy as np

# TODO: improve type hinting
class WFO:
    def update_config(self, config: Dict, split: Dict, balance_and_pos: Optional[Dict] = None):
        if balance_and_pos:
            config.update(balance_and_pos)

        config.update(
            {
                "start_date": ts_to_date((split["start_ms"] + self.ts_start) / 1000),
                "end_date": ts_to_date((split["end_ms"] + self.ts_start) / 1000),
                "n_days": split["diff_days"],
            }
        )

    def __init__(
        self,
        ticks: NDArray[(Any, 3), Float64],
        bc: dict,
        P_train: float = 0.2,
        P_test: float = 0.1,
        P_gap: float = 0.0,
        verbose: bool = True,
    ):
        self.step = {"train": P_train, "test": P_test, "gap": P_gap}
        self.ticks = ticks
        self.ts_start = ticks[0][2]
        self.ts_end = ticks[-1][2]
        self.ts_diff = self.ts_end - self.ts_start
        self.timeframe = self.diff(0, ticks.shape[0] - 1)
        self.bc = deepcopy(bc)
        self.verbose = verbose

    def __iter__(self) -> Iterator[Dict[str, Dict[str, float]]]:
        if self.verbose:
            return map(
                lambda x: {
                    "train": self.stats(**x["train"]),
                    "test": self.stats(**x["test"]),
                },
                self.chunks(),
            )
        else:
            return self.chunks()

    def backtest(self, config):
        results = []
        all_daily_returns = []
        all_objectives = []

        for splits in self:
            start, end = splits["train"]["start_idx"], splits["train"]["end_idx"]
            self.update_config(config, splits["train"])  # Update n_days and start/end date
            fills, stats, did_finish = backtest(config, self.ticks[start:end])
            _,_,result_ = analyze_backtest(fills, stats, self.bc)
            results.append(result_)

            all_daily_returns.append(result_["returns_daily" + "_obj"])  # stats is more accurate than fills
            all_objectives.append(result_[self.bc["metric"] + "_obj"])

        result = {}
        for k in results[0]:
            try:
                result[k] = np.mean([r[k] for r in results])
            except:
                result[k] = results[0][k]

        # Geometrical mean is often used to average returns
        result["daily_gains_gmean"] = np.exp(np.mean(np.log(np.array(all_daily_returns) + 1)))
        result["objective_gmean"] = np.exp(np.mean(np.log(np.array(all_objectives) + 1))) - 1

        return result

    def run(self):
        bc = deepcopy(self.bc)

        balance_and_pos = {
            "starting_balance": bc["starting_balance"],
            "long_pprice": 0.0,
            "long_psize": 0.0,
            "shrt_pprice": 0.0,
            "shrt_psize": 0.0,
        }

        all_daily_returns = []

        for k, split in enumerate(self):
            train = split["train"]
            test = split["test"]

            print("*** STARTIN BALANCE", balance_and_pos["starting_balance"])

            self.update_config(bc, train, balance_and_pos)

            analysis = backtest_tune(self.ticks[train["start_idx"] : train["end_idx"]], bc)
            candidate = clean_result_config(analysis.best_config)

            self.update_config(bc, test)
            fills, stats, did_finish = backtest(candidate, self.ticks[test["start_idx"] : test["end_idx"]])

            _,_,result = analyze_backtest(
                fills,
                stats,
                bc,
            )

            # Update the balance and positions with the last filled values of the testing run
            balance_and_pos = {key: fills[-1][key] for key in (balance_and_pos.keys() & fills[-1].keys())}
            balance_and_pos["starting_balance"] = stats[-1]["balance"]  # Same as fills

            all_daily_returns.append(result["returns_daily"])  # stats is more accurate than fills
            print("*** EQUITY", stats[-1]["equity"], all_daily_returns, "\n")

            # json.dump: candidate, result, stats, fills
            # we can load these files and generate plots/reports later
            #  with open(f"/tmp/candidate_test_{k}.json", "w") as f:
            #      json.dump(candidate, f)
            #  with open(f"/tmp/res_test_{k}.json", "w") as f:
            #      json.dump(result, f)
            #  with open(f"/tmp/stats_test_{k}.json", "w") as f:
            #      json.dump(stats, f)
            #  with open(f"/tmp/fills_test_{k}.json", "w") as f:
            #      json.dump(fills, f)

        returns_gmean = np.exp(np.mean(np.log(np.array(all_daily_returns) + 1))) - 1
        print("Geometrical mean of all the daily returns", returns_gmean)

    def chunks(self) -> Iterator[Dict[str, Dict[str, int]]]:
        for P_train_cur in np.arange(0.0, 1.0 - self.step["train"], self.step["test"]):
            train_idx, test_idx = {}, {}
            train_idx["start"] = self.find_tick_from_pct(P_train_cur)
            train_idx["end"] = self.find_tick_from_pct(P_train_cur + self.step["train"] - self.step["gap"])
            test_idx["start"] = self.find_tick_from_pct(P_train_cur + self.step["train"])
            test_idx["end"] = self.find_tick_from_pct(P_train_cur + self.step["train"] + self.step["test"])
            yield {
                "train": {
                    "start_idx": self.find_tick_from_pct(P_train_cur),
                    "end_idx": self.find_tick_from_pct(P_train_cur + self.step["train"] - self.step["gap"]),
                },
                "test": {
                    "start_idx": self.find_tick_from_pct(P_train_cur + self.step["train"]),
                    "end_idx": self.find_tick_from_pct(P_train_cur + self.step["train"] + self.step["test"]),
                },
            }

    def set_train_N(self, N: int) -> WFO:
        self.step["test"] = (1.0 - self.step["train"]) / float(N)
        self.step["gap"] = 0.0
        return self

    def set_step(self, key: str, x: float, unit: str) -> WFO:
        ts = self.convert(x, from_ts=False)[unit]
        self.step[key] = self.convert(ts)["pct"]
        return self

    def stats(self, start_idx: int, end_idx: int) -> Dict[str, float]:
        nested = {
            "start": self.diff(0, start_idx),
            "end": self.diff(0, end_idx),
            "diff": self.diff(start_idx, end_idx),
        }
        assert isinstance((ret := nested_to_record(nested, sep="_")), Dict)
        return ret

    def find_tick_from_pct(self, pct: float) -> Any:
        end_ts = self.ts_start + pct * self.timeframe["ms"]
        return min(self.timeframe["idx"], np.searchsorted(self.ticks[:, 2], end_ts))

    def convert(self, diff: float, from_ts: bool = True) -> Dict[str, float]:
        OP = div if from_ts else mul
        return {  # TODO: add pct and idx diff
            "ms": diff,
            "seconds": OP(diff, 1000.0),
            "minutes": OP(diff, (1000.0 * 60.0)),
            "hours": OP(diff, (1000.0 * 3600.0)),
            "days": OP(diff, (1000.0 * 3600.0 * 24.0)),
            "pct": OP(diff, self.ts_diff),
        }

    def diff(self, idx_start: int, idx_end: int) -> Dict[str, float]:
        return {
            "idx": idx_end - idx_start,
            **self.convert(self.ticks[idx_end][2] - self.ticks[idx_start][2]),
        }
