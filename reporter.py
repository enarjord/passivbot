import json
import os
import sys
from typing import Dict, List, Optional, Union

import pandas as pd
from ray.tune import CLIReporter
from ray.tune.progress_reporter import memory_debug_str, trial_errors_str, _get_trials_by_state
from ray.tune.trial import Trial
from ray.tune.utils import unflattened_lookup
from tabulate import tabulate

try:
    from collections.abc import Mapping, MutableMapping
except ImportError:
    from collections import Mapping, MutableMapping


def _get_trial_info(trial: Trial, parameters: List[str], metrics: List[str]):
    """
    Returns the following information about a trial:
    params... | metrics...
    @param trial: Trial to get information for.
    @param parameters: List of names of trial parameters to include.
    @param metrics: List of names of metrics to include.
    @return: List of column values.
    """
    result = trial.last_result
    config = trial.config
    trial_info = []
    trial_info += [unflattened_lookup(param, config, default=None) for param in parameters]
    trial_info += [unflattened_lookup(metric, result, default=None) for metric in metrics]
    return trial_info


def _get_trials_by_order(trials: List[Trial], metric: str, max_trials: int):
    """
    Sorts trials by metric function and discards low performing ones.
    @param trials: List of trials.
    @param metric: Metric to use.
    @param max_trials: Maximum number of trials to return.
    @return: List of best performing trials.
    """
    trials_by_order = sorted(trials, key=lambda k: k.last_result[metric] if metric in k.last_result else -1,
                             reverse=True)
    if not max_trials == float('inf'):
        trials_by_order = trials_by_order[:max_trials]
    return trials_by_order


def trial_progress_str(trials: List[Trial], metric: str, metric_columns: Union[List[str], Dict[str, str]],
                       parameter_columns: Union[None, List[str], Dict[str, str]] = None, total_samples: int = 0,
                       fmt: str = "psql", max_rows: Optional[int] = None, ):
    """
    Returns a human readable message for printing to the console.
    This contains a table where each row represents a trial, its parameters
    and the current values of its metrics.
    @param trials: List of trials to get progress string for.
    @param metric: Metric to use.
    @param metric_columns: Names of metrics to include. If this is a dict, the keys are metric names and the values are
    the names to use in the message. If this is a list, the metric name is used in the message directly.
    @param parameter_columns: Names of parameters to include. If this is a dict, the keys are parameter names and the
    values are the names to use in the message. If this is a list, the parameter name is used in the message directly.
    If this is empty, all parameters are used in the message.
    @param total_samples: Total number of trials that will be generated.
    @param fmt: Output format (see tablefmt in tabulate API).
    @param max_rows: Maximum number of rows in the trial table. Defaults to unlimited.
    @return: String to print.
    """
    messages = []
    delim = "<br>" if fmt == "html" else "\n"
    if len(trials) < 1:
        return delim.join(messages)

    num_trials = len(trials)
    trials_by_state = _get_trials_by_state(trials)

    num_trials_strs = ["{} {}".format(len(trials_by_state[state]), state) for state in sorted(trials_by_state)]

    if total_samples and total_samples >= sys.maxsize:
        total_samples = "infinite"

    messages.append("Number of trials: {}{} ({})".format(num_trials, f"/{total_samples}" if total_samples else "",
                                                         ", ".join(num_trials_strs)))

    messages += trial_progress_table(trials, metric, metric_columns, parameter_columns, fmt, max_rows)

    return delim.join(messages)


def trial_progress_table(trials: List[Trial], metric: str, metric_columns: Union[List[str], Dict[str, str]],
                         parameter_columns: Union[None, List[str], Dict[str, str]] = None, fmt: str = "psql",
                         max_rows: Optional[int] = None):
    """
    Create table view for trials.
    @param trials: List of trials to get progress table string for.
    @param metric: Metric to use.
    @param metric_columns: Names of metrics to include. If this is a dict, the keys are metric names and the values are
    the names to use in the message. If this is a list, the metric name is used in the message directly.
    @param parameter_columns: Names of parameters to include. If this is a dict, the keys are parameter names and the
    values are the names to use in the message. If this is a list, the parameter name is used in the message directly.
    If this is empty, all parameters are used in the message.
    @param fmt: Output format (see tablefmt in tabulate API).
    @param max_rows: Maximum number of rows in the trial table. Defaults to unlimited.
    @return: List of messages/rows.
    """
    messages = []
    num_trials = len(trials)

    max_rows = max_rows or float("inf")
    if num_trials > max_rows:
        trials = _get_trials_by_order(trials, metric, max_rows)
        overflow = num_trials - max_rows
    else:
        overflow = False
        trials = _get_trials_by_order(trials, metric, max_rows)

    if isinstance(metric_columns, Mapping):
        metric_keys = list(metric_columns.keys())
    else:
        metric_keys = metric_columns

    metric_keys = [k for k in metric_keys if
                   any(unflattened_lookup(k, t.last_result, default=None) is not None for t in trials)]

    if not parameter_columns:
        parameter_keys = sorted(set().union(*[t.evaluated_params for t in trials]))
    elif isinstance(parameter_columns, Mapping):
        parameter_keys = list(parameter_columns.keys())
    else:
        parameter_keys = parameter_columns

    trial_table = [_get_trial_info(trial, parameter_keys, metric_keys) for trial in trials]

    if isinstance(metric_columns, Mapping):
        formatted_metric_columns = [metric_columns[k] for k in metric_keys]
    else:
        formatted_metric_columns = metric_keys
    if isinstance(parameter_columns, Mapping):
        formatted_parameter_columns = [
            parameter_columns[k] for k in parameter_keys
        ]
    else:
        formatted_parameter_columns = parameter_keys
    columns = (formatted_parameter_columns + formatted_metric_columns)

    messages.append(tabulate(trial_table, headers=columns, tablefmt=fmt, showindex=False))
    if overflow:
        messages.append("... {} more trials not shown".format(overflow))
    return messages


class LogReporter(CLIReporter):
    """
    Extend CLI reporter to add saving of intermediate configs and results.
    """

    def __init__(self, metric_columns: Union[None, List[str], Dict[str, str]] = None,
                 parameter_columns: Union[None, List[str], Dict[str, str]] = None, total_samples: Optional[int] = None,
                 max_progress_rows: int = 20, max_error_rows: int = 20, max_report_frequency: int = 5,
                 infer_limit: int = 3, print_intermediate_tables: Optional[bool] = None, metric: Optional[str] = None,
                 mode: Optional[str] = None):
        self.objective = 0

        super(LogReporter, self).__init__(metric_columns, parameter_columns, total_samples, max_progress_rows,
                                          max_error_rows, max_report_frequency, infer_limit, print_intermediate_tables,
                                          metric, mode)

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        l = []
        o = []
        best_config = None
        best_eval = None
        config = None
        for trial in trials:
            if self._metric in trial.last_result:
                if trial.last_result[self._metric] > self.objective:
                    self.objective = trial.last_result[self._metric]
                    best_config = trial.config
                    best_eval = trial.evaluated_params
                l.append(trial.evaluated_params)
                o.append(trial.last_result[self._metric])
                config = trial.config

        try:
            if config:
                df = pd.DataFrame(l)
                df[self._metric] = o
                df.sort_values(self._metric, ascending=False, inplace=True)
                df.dropna(inplace=True)
                df[df[self._metric] > 0].to_csv(
                    os.path.join(config['session_dirpath'], 'intermediate_results.csv'), index=False)
                if best_eval:
                    intermediate_result = best_eval.copy()
                    intermediate_result['do_long'] = best_config['do_long']
                    intermediate_result['do_shrt'] = best_config['do_shrt']
                    json.dump(intermediate_result,
                              open(os.path.join(best_config['session_dirpath'], 'intermediate_best_result.json'), 'w'),
                              indent=4)
        except Exception as e:
            print("Something went wrong", e)

        print(self._progress_str(trials, done, *sys_info))

    def _progress_str(self, trials: List[Trial], done: bool, *sys_info: Dict, fmt: str = "psql", delim: str = "\n"):
        """
        Returns full progress string. This string contains a progress table and error table. The progress table
        describes the progress of each trial. The error table lists the error file, if any, corresponding to each trial.
        The latter only exists if errors have occurred.
        @param trials: Trials to report on.
        @param done: Whether this is the last progress report attempt.
        @param sys_info: System information.
        @param fmt: Table format. See `tablefmt` in tabulate API.
        @param delim: Delimiter between messages.
        @return: Table view of trials.
        """
        if not self._metrics_override:
            user_metrics = self._infer_user_metrics(trials, self._infer_limit)
            self._metric_columns.update(user_metrics)
        messages = ["== Status ==", memory_debug_str(), *sys_info]
        if done:
            max_progress = None
            max_error = None
        else:
            max_progress = self._max_progress_rows
            max_error = self._max_error_rows

        messages.append(trial_progress_str(trials, self._metric, metric_columns=self._metric_columns,
                                           parameter_columns=self._parameter_columns, total_samples=self._total_samples,
                                           fmt=fmt, max_rows=max_progress))
        messages.append(trial_errors_str(trials, fmt=fmt, max_rows=max_error))

        return delim.join(messages) + delim
