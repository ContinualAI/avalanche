Introduction
============

**Avalanche** is meant to provide a set of tools and resources for easily
prototype new continual learning algorithms and assess them in a comprehensive
way without effort. This can also help standardize training and evaluation
protocols in continual learning.

In order to achieve this goal the *avalanche* framework should be
general enough to quickly incorporate new CL strategies as well as new
benchmarks and metrics. While it would be great to be DL framework independent,
for simplicity I believe we should stick to Pytorch which today is becoming
the standard de-facto for machine learning research.

The framework is than split in three main modules:

    - [Benchmarks](avalanche/benchmarks): This module should maintain a uniform API for processing data in  a stream and contain all the major CL datasets / environments (similar to what has been done for Pytorch-vision).

    - [Training](avalanche/training): This module should provide all the utilities as well as a standard interface to implement and add a new continual learning strategy. All major CL baselines should be provided here.

    - [Evaluation](avalanche/evaluation): This modules should provide all the utilities and metrics that can help evaluate a CL strategy with respect to all the factors we think are important for CL.