import numpy as np

from avalanche._annotations import experimental


@experimental()
class MetricCollector:
    def __init__(self, stream):
        self.metrics_res = None

        self._stream_len = 0
        self._coeffs = []
        for exp in stream:
            self._stream_len += 1
            self._coeffs.append(len(exp.dataset))
        self._coeffs = np.array(self._coeffs)
        self._coeffs = self._coeffs / self._coeffs.sum()

    def update(self, res):
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
        assert time_reduce in {None}  # 'last', 'average'
        assert exp_reduce in {None, "sample_mean", "experience_mean"}
        assert name in self.metrics_res

        mvals = np.array(self.metrics_res[name])
        if exp_reduce is None:
            return mvals
        elif exp_reduce == "sample_mean":
            mvals = mvals * self._coeffs[None, :]  # weight by num.samples
            return mvals.sum(axis=1)  # weighted avg across exp.
        elif exp_reduce == "experience_means":
            return mvals.mean(axis=1)  # avg across exp.
        else:
            raise ValueError("BUG. It should never get here.")
