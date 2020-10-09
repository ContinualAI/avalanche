from avalanche.benchmarks import CMNIST
from avalanche.evaluation.metrics import *
from avalanche.extras.models import SimpleMLP
from avalanche.training.strategies import Naive
from avalanche.evaluation import EvalProtocol
from torch.utils.tensorboard import SummaryWriter

exp_name = "mnist_test"
log_dir = '../logs/' + exp_name
writer = SummaryWriter(log_dir)

# load the model with PyTorch for example
model = SimpleMLP()

# load the benchmark as a python iterator object
cdata = CMNIST()

# Eval Protocol
evalp = EvalProtocol(metrics=[ACC(), TimeUsage()])

# adding the CL strategy
clmodel = Naive(model, eval_protocol=evalp)

# getting full test set beforehand
test_full = cdata.get_full_testset()

results = []

# loop over the training incremental batches
for i, (x, y, t) in enumerate(cdata):

    # training over the batch
    print("Batch {0}, task {1}".format(i, t))
    clmodel.train(x, y, t)

    # here we could get the growing test set too
    # test_grow = cdata.get_growing_testset()

    # testing
    results.append(clmodel.test(test_full))
writer.close()
