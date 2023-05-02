import torchvision.transforms

from avalanche.training import SCR
import torch
import torchvision.transforms as transforms
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SlimResNet18, NCMClassifier, SCRModel
from avalanche.training.plugins import EvaluationPlugin

device = torch.device(
    f"cuda" if torch.cuda.is_available() else "cpu"
)
scenario = SplitCIFAR10(
    5,
    return_task_id=False,
    seed=0,
    fixed_class_order=list(range(10)),
    train_transform=transforms.ToTensor(),
    eval_transform=transforms.ToTensor(),
    shuffle=True,
    class_ids_from_zero_in_each_exp=False,
)

# SlimResNet18 is used as encoder
# the projection network takes as input the outpu of the ResNet
nf = 20
encoding_network = SlimResNet18(10, nf=nf)
encoding_network.linear = torch.nn.Identity()
projection_network = torch.nn.Sequential(torch.nn.Linear(nf*8, 128),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(128, 128))
# a NCM Classifier is used at eval time
model = SCRModel(
    feature_extractor=encoding_network,
    projection=projection_network)

optimizer = SGD(model.parameters(), lr=0.1)

interactive_logger = InteractiveLogger()
loggers = [interactive_logger]
training_metrics = []

# training accuracy cannot be monitored with SCR
# training loss and eval loss are two different ones
evaluation_metrics = [
    accuracy_metrics(stream=True),
    loss_metrics(epoch=True, stream=True),
]
evaluator = EvaluationPlugin(
    *training_metrics,
    *evaluation_metrics,
    loggers=loggers,
)

# around 48% final accuracy
cl_strategy = SCR(
    model,
    optimizer,
    augmentations=torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.RandomResizedCrop(size=32)
    ]),
    plugins=None,
    evaluator=evaluator,
    device=device,
    train_mb_size=10,
    eval_mb_size=64,
    mem_size=200,
    batch_size_mem=100
)
for t, experience in enumerate(scenario.train_stream):
    cl_strategy.train(experience)
    # cannot test on future experiences,
    # since NCM has no class means for unseen classes
    cl_strategy.eval(scenario.test_stream[:t+1])
