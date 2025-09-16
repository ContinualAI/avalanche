# IncDet‑EWC (Huber EWC + Pseudo‑Annotation)

## Paper summary
This paper presents IncDet, a framework that successfully adapts Elastic Weight Consolidation (EWC) to the task of incremental object detection. While EWC is effective in general incremental learning, it has previously been shown to fail when directly applied to object detection.

The authors identify two core issues responsible for this failure through controlled experiments:
1. Missing Old Class Annotations: When training on a new set of classes, images may contain objects from old classes that are not annotated. This causes the model to incorrectly learn to classify these old-class objects as background, leading to catastrophic forgetting.
2. Unstable Training: The quadratic regularisation loss used in EWC can cause gradient explosion when trying to balance performance between old and new classes, leading to unstable training.

To address these problems, the paper proposes two corresponding solutions:
1. Pseudo Annotation: To compensate for missing labels, the old model is used to predict bounding boxes for old-class objects in the new training images. These "pseudo" annotations are then combined with the ground-truth annotations for the new classes, preventing the model from misclassifying old objects as background.
2. Huber Regularization: A novel Huber regularization loss is introduced to replace EWC's original quadratic loss. This method adaptively clips the gradient for each parameter based on its importance to the old tasks, which prevents gradient explosion and allows for stable training and a better trade-off between remembering old classes and learning new ones.

These solutions are integrated into the IncDet framework, a general and flexible pipeline for incremental object detection. The process involves:
- Initial Training: A base model is trained on an initial set of classes.
- Predict & Aggregate: The trained model generates pseudo-annotations for old classes on new images, which are then aggregated with the manual annotations for the new classes.
- Incremental Fine-tuning: The model is fine-tuned using the combined annotations and the Huber regularization to learn the new classes while retaining knowledge of the old ones. This cycle can be executed recursively as more classes are added.

The framework was implemented using both Fast R-CNN and Faster R-CNN, demonstrating its versatility. Experiments on the PASCAL VOC and COCO datasets show that IncDet achieves new state-of-the-art results, surpassing previous methods in both final performance and in minimizing the performance gap compared to joint training on all classes. The proposed method is also more computationally and memory-efficient during training compared to prior auxiliary-based approaches.

Use `HuberEWCPlugin` with `PseudoAnnotationPlugin` to mitigate forgetting and leverage confident predictions on past classes.

## Quickstart (SplitMNIST)
```bash
python run_incdet_ewc.py --device auto --train_mb_size 64 --eval_mb_size 64 \
  --train_epochs 1 --lr 0.01 --momentum 0.9 --ewc_lambda 1000.0 --beta 0.5 \
  --confidence_thresh 0.95 --n_experiences 2 --seed 42
```

## CIFAR
```bash
python examples/incdet_ewc_cifar.py --benchmark cifar10 --use_huber 1 --train_mb_size 128 --train_epochs 2
```

## Minimal API
```python
from avalanche.training.plugins import HuberEWCPlugin, PseudoAnnotationPlugin

plugins = [
    HuberEWCPlugin(ewc_lambda=1000.0, beta=0.5),
    PseudoAnnotationPlugin(confidence_thresh=0.95),
]
# pass plugins=plugins into your Naive(...) strategy
```

## References
- IncDet: https://ieeexplore.ieee.org/document/9127478 — DOI: https://doi.org/10.1109/TNNLS.2020.3002583
- EWC: https://www.pnas.org/doi/10.1073/pnas.1611835114