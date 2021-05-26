"""
The :py:mod:`metrics` module provides a set of already
implemented metrics, ready to be used both standalone
and together with the `EvaluationPlugin`.
To use a standalone metric, please use the class which
inherits from `Metric` and manually call the appropriate
`update`, `reset` and 'result` method.
To automatically monitor metrics during training and evaluation
flows, specific classes which inherit from `PluginMetric` or
`GenericPluginMetric` are provided. Most of these metrics should
be instantiated by the user by leveraging
the related helper function. Such functions create an instance of
a specific metric (e.g. accuracy) and monitors it on multiple callbacks
(after each epoch, minibatch experience or stream).
For example, to print accuracy metrics at the
end of each training epoch and at the end of each evaluation experience,
it is only required to call `accuracy_metrics(epoch=True, experience=True)`
when creating the `EvaluationPlugin`.

When available, please always use helper functions to specify
the metrics to be monitored.

The following table describes all the metrics available in Avalanche.

=========================  ===================================================
Metric Name                Description
=========================  ===================================================
Top1_Acc                   The accuracy metric for single-label classification
Loss                       The specific loss is provided by
                           the user when creating the strategy.
Forgetting                 The difference between
                           the training performance and the evaluation
                           performance after training on future experiences.
Backward Transfer          The negative forgetting. That is, the difference
                           between the last evaluation performance and the
                           first training performance.
Confusion Matrix           A representation of
                           false/true positive/negatives for classification
Multiply and Accumulate    a.k.a. MAC. Estimates the computational cost
                           of the model forward pass on a single pattern.
                           Cost is estimated in terms of multiplications
                           operations.
Timing                     Time elapsed between different moments of the
                           execution
CPU Usage                  The average CPU consumption between different
                           moments of the execution.
RAM Usage                  The maximum RAM occupancy, as retrieved by
                           sampling its value at fixed intervals during
                           execution.
GPU Usage                  The maximum GPU occuapncy, as retrieved by
                           sampling its value at fixed intervals during
                           execution
Disk Usage                 The size in KB of the disk occupancy for a set
                           of file system paths.
=========================  ===================================================

The following table provides a brief description of when each metric can be
computed.
`Stream` specifies on which stream (training or evaluation) that metric
is computed.
Please, refer to the helper function of each metric to check which levels
are available for that metric.

=================  =================================================== ========
Level              Description                                         Stream
=================  =================================================== ========
MB (minibatch)     Metric emitted at the end of each training          Train
                   iteration. Its value is averaged across all
                   patterns in that minibatch. Metric is reset at the
                   beginning of each training iteration.
Epoch              Metric emitted at the end of each training epoch.   Train
                   Its value is averaged across all patterns in
                   that epoch. Metric is reset at the beginning of
                   each training epoch.
RunningEpoch       Metric emitted at the end of each training          Train
                   iteration. Its value is the average across all
                   patterns seen since the beginning of the epoch.
                   Metric is reset at the beginning of each
                   training epoch.
Experience         Metric emitted at the end of each evaluation        Eval
                   experience. Its value is averaged across all
                   patterns in that experience.
                   Metric is reset at the beginning of each
                   evaluation experience.
Stream             Metric emitted at the end of each evaluation        Eval
                   stream. Its value is averaged across all patterns
                   in that stream. Metric is reset at the beginning
                   of each evaluation stream.
=================  =================================================== ========

"""

from .mean import *
from .accuracy import *
from .confusion_matrix import *
from .cpu_usage import *
from .disk_usage import *
from .forgetting_bwt import *
from .gpu_usage import *
from .loss import *
from .mac import *
from .ram_usage import *
from .timing import *
