"""Ex-Model Continual Learning with Avalanche.

Ex-Model Continual Learning (ExML) is a CL scenario where the CL agent learns
from pre-trained expert models, while the raw data is unavailable (e.g. due to
privacy constraints).

See https://arxiv.org/abs/2112.06511 for more details.

Reference: Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021).
Ex-Model: Continual Learning from a Stream of Trained Models.
arXiv preprint arXiv:2112.06511.
"""
from torch.utils.data import DataLoader

from avalanche.benchmarks import ExMLMNIST
from avalanche.evaluation.metrics import Accuracy


if __name__ == "__main__":
    # ExML scenarios provide a stream of pretrained models
    exml_benchmark = ExMLMNIST(scenario="split")

    print(
        type(exml_benchmark).__name__,
        "testing expert models on the original train stream",
    )
    for i, model in enumerate(exml_benchmark.expert_models_stream):
        # Each model is trained on a separate experience of the training stream.
        # Here we simply check the accuracy on the training experience
        # for each expert model.
        # Notice that most models have a very high (train) accuracy because they
        # overfitted their own experience.

        model.to("cuda")
        acc = Accuracy()

        train_data = exml_benchmark.original_benchmark.train_stream[i].dataset
        for x, y, t in DataLoader(
            train_data, batch_size=256, pin_memory=True, num_workers=8
        ):
            x, y, t = x.to("cuda"), y.to("cuda"), t.to("cuda")
            y_pred = model(x)
            acc.update(y_pred, y, t)
        print(f"(i={i}) Original model accuracy: {acc.result()}")
        model.to("cpu")
