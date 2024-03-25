# TODO: doc
import json

import numpy as np

from avalanche._annotations import experimental


@experimental()
class MetricCollector:
    # TODO: doc
    def __init__(self, stream):
        # TODO: doc
        self.metrics_res = None

        self._stream_len = 0
        self._coeffs = []
        for exp in stream:
            self._stream_len += 1
            self._coeffs.append(len(exp.dataset))
        self._coeffs = np.array(self._coeffs)
        self._coeffs = self._coeffs / self._coeffs.sum()

    def update(self, res):
        # TODO: doc
        if self.metrics_res is None:
            self.metrics_res = {}
            for k, v in res.items():
                if len(v) != self._stream_len:
                    raise ValueError(f"Length does not correspond to stream. "
                                     f"Found {len(v)}, expected {self._stream_len}")
                self.metrics_res[k] = [v]
        else:
            for k, v in res.items():
                self.metrics_res[k].append(v)

    def get(self, name, time_reduce=None, exp_reduce=None):
        # TODO: doc
        assert time_reduce in {None, "last", "mean"}
        assert exp_reduce in {None, "sample_mean", "experience_mean"}
        assert name in self.metrics_res

        mvals = np.array(self.metrics_res[name])
        if exp_reduce is None:
            pass  # nothing to do here
        elif exp_reduce == "sample_mean":
            mvals = mvals * self._coeffs[None, :]  # weight by num.samples
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
