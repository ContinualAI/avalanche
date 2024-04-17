"""
Utilities for metrics collection outside of Avalanche training Templates.
"""

import json

import numpy as np

from avalanche._annotations import experimental


@experimental()
class MetricCollector:
    """A simple metric collector object.

    Functionlity includes the ability to store metrics over time and compute
    aggregated values of them. Serialization is supported via json files.

    You can pass a stream to the `update` method to compute separate metrics
    for each stream. Metric names will become `{stream.name}/{metric_name}`.
    When you recover the stream using the `get` method, you need to pass the
    same stream.

    Example usage:

    .. code-block:: python

        mc = MetricCollector()
        for exp in train_stream:
            res = {"Acc": 0.1} # metric dictionary for the current timestep
            mc.update(res, stream=test_stream)
        acc_timeline = mc.get("Accuracy", exp_reduce="sample_mean", stream=test_stream)

    """

    def __init__(self):
        """Init."""
        self.metrics_res = {}

        self._stream_len = {}  # stream-name -> stream length
        self._coeffs = {}  # stream-name -> list_of_exp_length

    def _init_stream(self, stream):
        # we compute the stream length and number of samples in each experience
        # only the first time we encounter the experience because
        # iterating over the stream may be expensive
        self._stream_len[stream.name] = 0
        coeffs = []
        for exp in stream:
            self._stream_len[stream.name] += 1
            coeffs.append(len(exp.dataset))
        coeffs = np.array(coeffs)
        self._coeffs[stream.name] = coeffs / coeffs.sum()

    def update(self, res, *, stream=None):
        """Update the metrics.

        :param res: a dictionary of new metrics with <metric_name: value> items.
        :param stream: optional stream. If a stream is given the full metric
            name becomes `f'{stream.name}/{metric_name}'`.
        """
        for k, v in res.items():
            # optional safety check on metric shape
            # metrics are expected to have one value for each experience in the stream
            if stream is not None:
                if stream.name not in self._stream_len:
                    self._init_stream(stream)
                if len(v) != self._stream_len[stream.name]:
                    raise ValueError(
                        f"Length does not correspond to stream. "
                        f"Found {len(v)}, expected {self._stream_len[stream.name]}"
                    )

            # update metrics dictionary
            if stream is not None:
                k = f"{stream.name}/{k}"
            if k in self.metrics_res:
                self.metrics_res[k].append(v)
            else:
                self.metrics_res[k] = [v]

    def get(
        self, name, *, time_reduce=None, exp_reduce=None, stream=None, weights=None
    ):
        """Returns a metric value given its name and aggregation method.

        :param name: name of the metric.
        :param time_reduce: Aggregation over the time dimension. One of {None, 'last', 'mean'}, where:
            - None (default) does not use any aggregation
            - 'last' returns the last timestep
            - 'mean' averages over time
        :param exp_reduce: Aggregation over the experience dimension. One of {None, 'sample_mean', 'experience_mean'} where:
            - None (default) does not use any aggregation
            - `sample_mean` is an average weighted by the number of samples in each experience
            - `experience_mean` is an experience average.
            - 'weighted_sum' is a weighted sum of the experiences using the `weights` argument.
        :param stream: stream that was used to compute the metric. This is
            needed to build the full metric name if the get was called with a
            stream name and if `exp_reduce == sample_mean` to get the number
            of samples from each experience.
        :param weights: weights for each experience when `exp_reduce == 'weighted_sum`.
        :return: aggregated metric value.
        """
        assert time_reduce in {None, "last", "mean"}
        assert exp_reduce in {None, "sample_mean", "experience_mean", "weighted_sum"}
        if exp_reduce == "weighted_sum":
            assert (
                weights is not None
            ), "You should set the `weights` argument when `exp_reduce == 'weighted_sum'`."
        else:
            assert (
                weights is None
            ), "Can't use the `weights` argument when `exp_reduce != 'weighted_sum'`"

        if stream is not None:
            name = f"{stream.name}/{name}"
        if name not in self.metrics_res:
            print(
                f"{name} metric was not found. Maybe you forgot the `stream` argument?"
            )

        mvals = np.array(self.metrics_res[name])
        if exp_reduce is None:
            pass  # nothing to do here
        elif exp_reduce == "sample_mean":
            if stream is None:
                raise ValueError(
                    "If you want to use `exp_reduce == sample_mean` you need to provide"
                    "the `stream` argument to the `update` and `get` methods."
                )
            if stream.name not in self._coeffs:
                self._init_stream(stream)
            mvals = mvals * self._coeffs[stream.name][None, :]  # weight by num.samples
            mvals = mvals.sum(axis=1)  # weighted avg across exp.
        elif exp_reduce == "experience_mean":
            mvals = mvals.mean(axis=1)  # avg across exp.
        elif exp_reduce == "weighted_sum":
            weights = np.array(weights)[None, :]
            mvals = (mvals * weights).sum(axis=1)
        else:
            raise ValueError("BUG. It should never get here.")

        if time_reduce is None:
            pass  # nothing to do here
        elif time_reduce == "last":
            mvals = mvals[-1]  # last timestep
        elif time_reduce == "mean":
            mvals = mvals.mean(axis=0)  # avg. over time
        else:
            raise ValueError("BUG. It should never get here.")

        return mvals

    def get_dict(self):
        """Returns metrics dictionary.

        :return: metrics dictionary
        """
        # TODO: test
        return self.metrics_res

    def load_dict(self, d):
        """Loads a new metrics dictionary.

        :param d: metrics dictionary
        """
        # TODO: test
        self.metrics_res = d

    def load_json(self, fname):
        """Loads a metrics dictionary from a json file.

        :param fname: file name
        """
        # TODO: test
        with open(fname, "w") as f:
            self.metrics_res = json.load(f)

    def to_json(self, fname):
        """Stores metrics dictionary as a json filename.

        :param fname:
        :return:
        """
        # TODO: test
        with open(fname, "w") as f:
            json.dump(obj=self.metrics_res, fp=f)
