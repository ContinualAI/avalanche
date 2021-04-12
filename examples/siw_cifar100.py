from avalanche.benchmarks.classic import SplitCIFAR100
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.strategies import BaseStrategy
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, \
    AGEMPlugin, SIWPlugin
import torchvision
from avalanche.benchmarks.generators import filelist_scenario, \
    dataset_scenario, tensor_scenario, paths_scenario
from torchvision.transforms import Compose, CenterCrop, Normalize, \
    Scale, Resize, ToTensor, ToPILImage
import torchvision.transforms as transforms
import torch.nn as nn
import torch as th
import torch.cuda as tc
from torch.autograd import Variable

################################################
P = 10  # number of classes in each state
device = 'cuda:0'
#############################################
siw = SIWPlugin(batch_size=32, num_workers=8)
model = torchvision.models.resnet18(num_classes=100).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = CrossEntropyLoss()
strategy = BaseStrategy(model, optimizer, criterion, plugins=[siw],
                        device=device, train_epochs=10)

normalize = transforms.Normalize(mean=[0.5356, 0.4898, 0.4255],
                                 std=[0.2007, 0.1999, 0.1992])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize])

# scenario
scenario = SplitCIFAR100(n_experiences=10, return_task_id=False,
                         seed=1234, train_transform=train_transform,
                         eval_transform=test_transform)

# TRAINING LOOP
print('Starting experiment...')
results = []
for i, experience in enumerate(scenario.train_stream):
    print("Start of experience: ", experience.current_experience)
    strategy.train(experience)
    print('Training completed')
    print('Computing accuracy on the test set')
    res = strategy.eval(scenario.test_stream[i])
    results.append(res)

print('Results = ' + str(results))
