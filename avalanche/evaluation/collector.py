# TODO: doc
import json

import numpy as np

from avalanche._annotations import experimental


@experimental()
class MetricCollector:
    # TODO: doc
    def __init__(self):
        # TODO: doc
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
        # TODO: doc
        # TODO: test multi groups
        for k, v in res.items():
            # optional safety check on metric shape
            # metrics are expected to have one value for each experience in the stream
            if stream is not None:
                if stream.name not in self._stream_len:
                    self._init_stream(stream)
                if len(v) != self._stream_len[stream.name]:
                    raise ValueError(f"Length does not correspond to stream. "
                                     f"Found {len(v)}, expected {self._stream_len[stream.name]}")

            # update metrics dictionary
            if stream is not None:
                k = f"{stream.name}/{k}"
            if k in self.metrics_res:
                self.metrics_res[k].append(v)
            else:
                self.metrics_res[k] = [v]

    def get(self, name, *, time_reduce=None, exp_reduce=None, stream=None):
        # TODO: doc
        assert time_reduce in {None, "last", "mean"}
        assert exp_reduce in {None, "sample_mean", "experience_mean"}

        if stream is not None:
            name = f"{stream.name}/{name}"
        if name not in self.metrics_res:
            print(f"{name} metric was not found. Maybe you forgot the `stream` argument?")

        mvals = np.array(self.metrics_res[name])
        if exp_reduce is None:
            pass  # nothing to do here
        elif exp_reduce == "sample_mean":
            if stream is None:
                raise ValueError(
                    "If you want to use `exp_reduce == sample_mean` you need to provide"
                    "the `stream` argument to the `update` and `get` methods.")
            if stream.name not in self._coeffs:
                self._init_stream(stream)
            mvals = mvals * self._coeffs[stream.name][None, :]  # weight by num.samples
            mvals = mvals.sum(axis=1)  # weighted avg across exp.
        elif exp_reduce == "experience_mean":
            mvals = mvals.mean(axis=1)  # avg across exp.
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
        # TODO: doc
        # TODO: test
        return self.metrics_res

    def load_dict(self, d):
        # TODO: doc
        # TODO: test
        self.metrics_res = d

    def load_json(self, fname):
        # TODO: doc
        # TODO: test
        with open(fname, 'w') as f:
            self.metrics_res = json.load(f)

    def to_json(self, fname):
        # TODO: doc
        # TODO: test
        with open(fname, 'w') as f:
            json.dump(obj=self.metrics_res, fp=f)
