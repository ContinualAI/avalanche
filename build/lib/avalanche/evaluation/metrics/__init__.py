"""
The :py:mod:`metrics` module provides a set of already
implemented metrics, ready to be used both standalone
and together with the `EvaluationPlugin`.
To use a standalone metric, please use the class which
inherits from `Metric` and manually call the appropriate
`update`, `reset` and 'result` method.
To automatically monitor metrics during training and evaluation
flows, specific classes which inherit from `PluginMetric`
are provided. Most of these metrics can be created by leveraging
the related helper function. Such function instantiates the same
metric monitored on multiple callbacks (after each epoch, minibatch
or experience). For example, to print accuracy metrics at the
end of each training epoch and at the end of each evaluation experience,
it is only required to call `accuracy_metrics(epoch=True, experience=True)`
when creating the `EvaluationPlugin`.

When available, please always use helper functions to specify
the metrics to be monitored.
"""

from .mean import *
from .accuracy import *
from .confusion_matrix import *
from .cpu_usage import *
from .disk_usage import *
from .forgetting import *
from .gpu_usage import *
from .loss import *
from .mac import *
from .ram_usage import *
from .timing import *
