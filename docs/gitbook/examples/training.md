---
description: Baselines and Strategies Code Examples
---

# Training

_Avalanche_ offers significant support for _training_ (with _templates_, _strategies_ and _plug-ins_). Here you can find a list of **examples** related to the training and some strategies available in Avalanche (each strategy reproduces original paper results in the [CL-Baselines](https://github.com/ContinualAI/continual-learning-baselines) repository:&#x20;

* [Joint-Training](../../../examples/joint\_training.py): _this example shows how to take a stream of experiences and train simultaneously on all of them. This is useful to implement the "offline"  or "multi-task" upper bound._
* [AR1 strategy](../../../examples/ar1.py): t_his is a simple example on how to use the AR1 strategy._
* [Cumulative Strategy](../../../examples/dataloader.py): h_ow to define your own cumulative strategy based on the different Data Loaders made available in Avalanche._&#x20;
* [Early Stopping](../../../examples/all\_mnist\_early\_stopping.py): _this example shows how to use early stopping to dynamically stop the training procedure when the model converged instead of training for a fixed number of epochs._
* [Object Detection](../../../examples/detection.py): _this example shows how to run object detection/segmentation tasks._
* [Object Detection with Elvis](../../../examples/detection\_lvis.py)_: this example shows how to run object detection/segmentation tasks with a_ _toy benchmark based on the LVIS dataset._
* [Object Detection Training](https://github.com/ContinualAI/avalanche/tree/master/examples/tvdetection): _set of examples showing how you can use Avalanche for distributed training of object detector._
* [Ex-Model Continual Learning](../../../examples/ex\_model\_cl.py)_: this example shows how to create a stream of pre-trained model from which to learn._
* [Generative Replay](../../../examples/generative\_replay\_MNIST\_generator.py)_: this is a simple example on how to implement generative replay in Avalanche._
* [Online Naive](https://github.com/ContinualAI/avalanche/blob/6dbabb2ab787a53b59b9cbcb245ad500e984f671/examples/online\_naive.py): _example to run a naive strategy in an online setting._
* [Continual Sequence Classification](../../../examples/continual\_sequence\_classification.py): _sequence classification example using torchaudio and Speech Commands._
