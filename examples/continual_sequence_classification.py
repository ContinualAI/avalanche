try:
    import torchaudio
except ImportError:
    raise ModuleNotFoundError(
        "TorchAudio package is required to load its dataset. "
        "You can install it as extra dependency with "
        "`pip install avalanche-lib[extra]`"
    )
import torch
import avalanche as avl
from avalanche.benchmarks.datasets.torchaudio_wrapper import SpeechCommands
from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised.strategy_wrappers import Naive


def main():
    n_exp = 7  # 7 experiences -> 5 classes per experience
    hidden_rnn_size = 32
    lr = 1e-3
    # WARNING: Enabling MFCC greatly slows down the runtime execution
    mfcc = False

    if mfcc:
        mfcc_preprocess = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, 
            melkwargs={"n_mels": 50, "hop_length": 10}
        )
    else:
        mfcc_preprocess = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SpeechCommands(
        subset="training", 
        mfcc_preprocessing=mfcc_preprocess
    )
    test_ds = SpeechCommands(
        subset="testing",  # you may also use "validation"
        mfcc_preprocessing=mfcc_preprocess,
    )

    benchmark = nc_benchmark(
        train_dataset=train_ds,
        test_dataset=test_ds,
        shuffle=True,
        train_transform=None,
        eval_transform=None,
        n_experiences=n_exp,
        task_labels=False,
    )

    classes_in_experience = [
        benchmark.classes_in_experience["train"][i]
        for i in range(benchmark.n_experiences)
    ]
    print(f"Number of training experiences: {len(benchmark.train_stream)}")
    print(f"Number of test experiences: {len(benchmark.test_stream)}")
    print(f"Number of classes: {benchmark.n_classes}")
    print(f"Classes per experience: " f"{classes_in_experience}")

    input_size = 1 if mfcc_preprocess is None else mfcc_preprocess.n_mfcc
    model = avl.models.SimpleSequenceClassifier(
        input_size=input_size,
        hidden_size=hidden_rnn_size,
        n_classes=benchmark.n_classes,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            epoch=True, experience=True, stream=True
        ),
        loss_metrics(
            epoch=True, experience=True, stream=True
        ),
        loggers=[InteractiveLogger()]
    )

    strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=64,
        train_epochs=1,
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin
    )

    for exp in benchmark.train_stream:
        strategy.train(exp)
        strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    main()
