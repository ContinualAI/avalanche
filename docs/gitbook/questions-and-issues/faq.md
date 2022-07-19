---
description: Frequently Asked Questions
---

# FAQ

In this page we answer _frequently asked questions_ about the library. We know these to be mostly pain points we need to address as soon as possible in the form of better features o better documentation.

> How can I create a stream of experiences based on my own data?

You can use the [Benchmark Generators](https://avalanche-api.continualai.org/en/v0.1.0/benchmarks.html#benchmark-generators): such utils in Avalanche allows you to build a stream of experiences based on an AvalancheDataset (or PyTorchDataset), or directly from PyTorch tensors, paths or filelists.&#x20;

> Why some Avalanche strategies do not work on my dataset?

We cannot guarantee each strategy implemented in Avalanche will work in any possible setting. A continual learning algorithm implementation is accepted in Avalanche if it can reproduce at least a portion of the original paper results. In the [CL-Baseline](https://github.com/ContinualAI/reproducible-continual-learning) project we make sure reproducibility is maintained for those with every main avalanche release.&#x20;
