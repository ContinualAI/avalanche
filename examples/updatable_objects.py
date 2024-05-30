import os

import setGPU

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from avalanche.benchmarks import SplitMNIST, with_classes_timeline
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.core import Agent

from avalanche.evaluation.collector import MetricCollector
from avalanche.evaluation.functional import forgetting
from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation.plot_utils import plot_metric_matrix
from avalanche.models import SimpleMLP, IncrementalClassifier
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from avalanche.models.dynamic_optimizers import update_optimizer, DynamicOptimizer
from avalanche.training import ReservoirSamplingBuffer, LearningWithoutForgetting
from avalanche.training.losses import MaskedCrossEntropy


def train_experience(agent_state, exp, epochs=10):
    agent_state.model.train()
    # avalanche datasets have train/eval modes to switch augmentations
    data = exp.dataset.train()
    agent_state.pre_adapt(exp)  # update objects and call pre_hooks
    for ep in range(epochs):
        if len(agent_state.replay.buffer) > 0:
            # if the replay buffer is not empty we sample from
            # current data and replay buffer in parallel
            dl = ReplayDataLoader(
                data, agent_state.replay.buffer, batch_size=32, shuffle=True
            )
        else:
            dl = DataLoader(data, batch_size=32, shuffle=True)

        for x, y, _ in dl:
            x, y = x.cuda(), y.cuda()
            agent_state.opt.zero_grad()
            yp = agent_state.model(x)
            l = agent_state.loss(yp, y)

            # you may have to change this if your regularization loss
            # needs different arguments
            l += agent_state.reg_loss(x, yp, agent_state.model)

            l.backward()
            agent_state.opt.step()
    agent_state.post_adapt(exp)  # update objects and call post_hooks


@torch.no_grad()
def my_eval(model, stream, metrics):
    """Evaluate `model` on `stream` computing `metrics`.

    Returns a dictionary {metric_name: list-of-results}.
    """
    model.eval()
    res = {uo.__class__.__name__: [] for uo in metrics}
    for exp in stream:
        [uo.reset() for uo in metrics]
        dl = DataLoader(exp.dataset.eval(), batch_size=512, num_workers=8)
        for x, y, _ in dl:
            x, y = x.cuda(), y.cuda()
            yp = model(x)
            [uo.update(yp, y) for uo in metrics]
        [res[uo.__class__.__name__].append(uo.result()) for uo in metrics]
    return res


if __name__ == "__main__":
    bm = SplitMNIST(n_experiences=5)
    train_stream, test_stream = bm.train_stream, bm.test_stream
    # we split the training stream into online experiences
    ocl_stream = split_online_stream(train_stream, experience_size=256, seed=1234)
    # we add class attributes to the experiences
    ocl_stream = with_classes_timeline(ocl_stream)

    # agent state collects all the objects that are needed during training
    # many of these objects will have some state that is updated during training.
    # Put all the stateful training objects here. Calling the agent `pre_adapt`
    # and `post_adapt` methods will adapt all the objects.
    # Sometimes, it may be useful to also put non-stateful objects (e.g. losses)
    # to easily switch between them while keeping the same training loop.
    agent = Agent()
    agent.replay = ReservoirSamplingBuffer(max_size=200)
    agent.loss = MaskedCrossEntropy()
    agent.reg_loss = LearningWithoutForgetting(alpha=1, temperature=2)

    # Avalanche models support dynamic addition and expansion of parameters
    agent.model = SimpleMLP().cuda()
    agent.model.classifier = IncrementalClassifier(in_features=512).cuda()
    agent.add_pre_hooks(lambda a, e: avalanche_model_adaptation(a.model, e))

    # optimizer and scheduler
    # we have update the optimizer before each experience.
    # This is needed because the model's parameters may change if you are using
    # a dynamic model.
    opt = SGD(agent.model.parameters(), lr=0.001)
    agent.opt = DynamicOptimizer(opt)
    agent.scheduler = ExponentialLR(opt, gamma=0.999)
    # we use a hook to call the scheduler.
    # we update the lr scheduler after each experience (not every epoch!)
    agent.add_post_hooks(lambda a, e: a.scheduler.step())

    print("Start training...")
    metrics = [Accuracy()]
    mc = MetricCollector()  # we put all the metric values here
    for exp in ocl_stream:
        if exp.logging().is_first_subexp:  # new drift
            print("training on classes ", exp.classes_in_this_experience)
        train_experience(agent, exp, epochs=2)
        if exp.logging().is_last_subexp:
            # after learning new classes, do the eval
            # notice that even in online settings we use the non-online stream
            # for evaluation because each experience in the original stream
            # corresponds to a separate data distribution.
            res = my_eval(agent.model, train_stream, metrics)
            mc.update(res, stream=train_stream)
            print("TRAIN METRICS: ", res)
            res = my_eval(agent.model, test_stream, metrics)
            mc.update(res, stream=test_stream)
            print("TEST METRICS: ", res)
            print()

    os.makedirs("./logs", exist_ok=True)
    torch.save(agent.model.state_dict(), "./logs/model.pth")

    print(mc.get_dict())
    mc.to_json("./logs/mc.json")  # save metrics

    acc_timeline = mc.get("Accuracy", exp_reduce="sample_mean", stream=test_stream)
    print("Accuracy:\n", mc.get("Accuracy", exp_reduce=None, stream=test_stream))
    print("AVG Accuracy: ", acc_timeline)

    acc_matrix = mc.get("Accuracy", exp_reduce=None, stream=test_stream)
    fig = plot_metric_matrix(acc_matrix, title="Accuracy - Train")
    fig.savefig("./logs/accuracy.png")

    fm = forgetting(acc_matrix)
    print("Forgetting:\n", fm)
    fig = plot_metric_matrix(fm, title="Forgetting - Train")
    fig.savefig("./logs/forgetting.png")

    # BONUS: example of re-loading the model from disk
    model = SimpleMLP()
    model.classifier = IncrementalClassifier(in_features=512)
    for exp in ocl_stream:  # we need to adapt the model's architecture again
        avalanche_model_adaptation(model, exp)
    model.load_state_dict(torch.load("./logs/model.pth"))
