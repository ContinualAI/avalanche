---
description: Examples for the Models module offered in Avalanche
---

# Models

_Avalanche_ offers basic support for defining your own models or adapt existing PyTorch models with a particular emphasis on model adaptation over time.&#x20;

You can find **examples** related to the models here:&#x20;

* [Using PyTorchCV pre-trained models](../../../examples/pytorchcv\_models.py): _This example shows how to train models provided by pytorchcv with the rehearsal strategy._
* [Use a Multi-Head model](https://github.com/ContinualAI/avalanche/blob/6dbabb2ab787a53b59b9cbcb245ad500e984f671/examples/multihead.py#L27): _This example trains a Multi-head model on Split MNIST with Elastich Weight Consolidation. Each experience has a different task label, which is used at test time to select the appropriate head._&#x20;

