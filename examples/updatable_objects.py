from types import SimpleNamespace

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from avalanche.benchmarks import SplitMNIST, with_classes_timeline
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

from avalanche.evaluation.collector import MetricCollector
from avalanche.evaluation.functional import forgetting
from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation.plot_utils import plot_metric_matrix
from avalanche.models import SimpleMLP
from avalanche.training import ReservoirSamplingBuffer
from avalanche.training.losses import MaskedCrossEntropy


def train_experience(agent_state, exp, epochs=10):
    # defining the training for a single exp. instead of the whole stream
    # should be better because this way training does not need to care about eval loop.
    # agent_state takes the role of the strategy object

    # this is the usual before_exp
    # updatable_objs = [agent_state.replay, agent_state.loss, agent_state.model]
    # [uo.pre_update(exp, agent_state) for uo in updatable_objs]
    agent_state.loss.before_training_exp(SimpleNamespace(experience=exp))
    # agent_state.model.recursive_adaptation(exp)
    # agent_state.pre_update(agent_state, exp)

    data = exp.dataset.train()
    for ep in range(epochs):
        agent_state.model.train()
        if len(agent_state.replay.buffer) > 0:
            dl = ReplayDataLoader(data, agent_state.replay.buffer,
                                  batch_size=32, shuffle=True)
        else:
            dl = DataLoader(data, batch_size=32, shuffle=True)

        for x, y, _ in dl:
            x, y = x.cuda(), y.cuda()
            agent_state.opt.zero_grad()
            yp = agent_state.model(x)
            l = agent_state.loss(yp, y)
            # l += agent_state.reg_loss()
            l.backward()
            agent_state.opt.step()

        # this is the usual after_exp
        # updatable_objs = [agent_state.replay, agent_state.loss, agent_state.model]
        # [uo.post_update(exp, agent_state) for uo in updatable_objs]
        agent_state.replay.update(SimpleNamespace(experience=exp))
    agent_state.scheduler.step()
    # agent_state.post_update(agent_state, exp)
    return agent_state


@torch.no_grad()
def my_eval(model, stream, metrics, **extra_args):
    # eval also becomes simpler. Notice how in Avalanche it's harder to check whether
    # we are evaluating a single exp. or the whole stream.
    # Now we evaluate each stream with a separate function call

    res = {uo.__class__.__name__: [] for uo in metrics}
    for exp in stream:
        [uo.reset() for uo in metrics]
        dl = DataLoader(exp.dataset, batch_size=512, num_workers=8)
        for x, y, _ in dl:
            x, y = x.cuda(), y.cuda()
            yp = model(x)
            [uo.update(yp, y) for uo in metrics]
        [res[uo.__class__.__name__].append(uo.result()) for uo in metrics]
    return res


if __name__ == '__main__':
    bm = SplitMNIST(n_experiences=5)
    train_stream, test_stream = bm.train_stream, bm.test_stream
    ocl_stream = split_online_stream(train_stream, experience_size=256, seed=1234)
    ocl_stream = with_classes_timeline(ocl_stream)

    # agent state collects all the objects that are needed during training
    # many of these objects will have some state that is updated during training.
    # The training function returns the updated agent state at each step.
    agent_state = SimpleNamespace()
    agent_state.replay = ReservoirSamplingBuffer(max_size=200)
    agent_state.loss = MaskedCrossEntropy()

    # model
    agent_state.model = SimpleMLP(num_classes=10).cuda()

    # optim and scheduler
    agent_state.opt = SGD(agent_state.model.parameters(), lr=0.001)
    agent_state.scheduler = ExponentialLR(agent_state.opt, gamma=0.999)

    print("Start training...")
    metrics = [Accuracy()]
    mc_test = MetricCollector(test_stream)
    for exp in ocl_stream:
        print(exp.classes_in_this_experience, end=' ')
        agent_state = train_experience(agent_state, exp, epochs=2)
        if exp.logging().is_last_subexp:
            print()
            res = my_eval(agent_state.model, test_stream, metrics)
            print("EVAL: ", res)
            mc_test.update(res)

    acc_timeline = mc_test.get("Accuracy", exp_reduce="sample_mean")
    print(mc_test.get("Accuracy", exp_reduce=None))
    print(acc_timeline)

    acc_matrix = mc_test.get("Accuracy", exp_reduce=None)
    fig = plot_metric_matrix(acc_matrix, title="Accuracy - Train")
    fig.savefig("accuracy.png")

    fm = forgetting(acc_matrix)
    fig = plot_metric_matrix(fm, title="Forgetting - Train")
    fig.savefig("forgetting.png")
