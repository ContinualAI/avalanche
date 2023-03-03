---
description: Baselines and Strategies Code Examples
---

# Training

_Avalanche_ offers significant support for _training_ (with _templates_, _strategies_ and _plug-ins_). Here you can find a list of **examples** related to the training and some strategies available in Avalanche (each strategy reproduces original paper results in the [CL-Baselines](https://github.com/ContinualAI/continual-learning-baselines) repository:&#x20;

* [Joint-Training](../../../examples/joint\_training.py): _this example shows how to take a stream of experiences and train simultaneously on all of them. This is useful to implement the "offline"  or "multi-task" upper bound._
* [Replay strategy](../../../examples/replay.py)_: simple example on the usage of replay in Avalanche._
* [AR1 strategy](../../../examples/ar1.py): t_his is a simple example on how to use the AR1 strategy._
* [CoPE Strategy](../../../examples/cope.py): _this is a simple example on how to use the CoPE plugin. It's an example in the online data incremental setting, where both learning and evaluation is completely task-agnostic._
* [Cumulative Strategy](../../../examples/dataloader.py): h_ow to define your own cumulative strategy based on the different Data Loaders made available in Avalanche._&#x20;
* [Deep SLDA](../../../examples/deep\_slda.py)_: this is a simple example on how to use the Deep SLDA strategy._
* [Early Stopping](../../../examples/all\_mnist\_early\_stopping.py): _this example shows how to use early stopping to dynamically stop the training procedure when the model converged instead of training for a fixed number of epochs._
* [Object Detection](../../../examples/detection.py): _this example shows how to run object detection/segmentation tasks._
* [Object Detection with Elvis](../../../examples/detection\_lvis.py)_: this example shows how to run object detection/segmentation tasks with a_ _toy benchmark based on the LVIS dataset._
* [Object Detection Training](https://github.com/ContinualAI/avalanche/tree/master/examples/tvdetection): _set of examples showing how you can use Avalanche for distributed training of object detector._
* [EWC on MNIST](../../../examples/ewc\_mnist.py)_: this example tests EWC on Split MNIST and Permuted MNIST._
* [LWF on MNIST](../../../examples/lfl\_mnist.py)_: this example tests LWF on Permuted MNIST._
* [GEM and A-GEM on MNIST](../../../examples/gem\_agem\_mnist.py)_: this example shows how to use GEM and A-GEM strategies on MNIST._
* [Ex-Model Continual Learning](../../../examples/ex\_model\_cl.py)_: this example shows how to create a stream of pre-trained model from which to learn._
* [Generative Replay](../../../examples/generative\_replay\_MNIST\_generator.py)_: this is a simple example on how to implement generative replay in Avalanche._
* [iCARL strategy](../../../examples/icarl.py): _simple example to show how to use the iCARL strategy._
* [LaMAML strategy](../../../examples/lamaml\_cifar100.py)_: example on how to use a meta continual learning in Avalanche._
* [RWalk strategy](../../../examples/rwalk\_mnist.py): _example of the RWalk strategy usage._
* [Online Naive](https://github.com/ContinualAI/avalanche/blob/6dbabb2ab787a53b59b9cbcb245ad500e984f671/examples/online\_naive.py): _example to run a naive strategy in an online setting._
* [Synaptic Intelligence](../../../examples/synaptic\_intelligence.py): _this is a simple example on how to use the Synaptic Intelligence Plugin._
* [Continual Sequence Classification](../../../examples/continual\_sequence\_classification.py): _sequence classification example using torchaudio and Speech Commands._
