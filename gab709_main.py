from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.strategies import GSS_greedy, Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.benchmarks.generators.benchmark_generators import data_incremental_benchmark
from avalanche.evaluation.metrics import ExperienceForgetting, accuracy_metrics,loss_metrics, timing_metrics,cpu_usage_metrics, StreamConfusionMatrix,disk_usage_metrics, gpu_usage_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger

model = SimpleMLP(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

# scenario
scenario = data_incremental_benchmark(SplitMNIST(n_experiences=5, seed=1),experience_size=500)

#Logging

tb_logger = TensorboardLogger()

text_logger = TextLogger(open('log.txt', 'a'))

interactive_logger = InteractiveLogger()


eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    ExperienceForgetting(),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger]
)


cl_strategy = GSS_greedy(
    model, optimizer, criterion, 
    train_mb_size=100, mem_strength=5, input_size=[1, 28, 28], train_epochs=2, eval_mb_size=100, mem_size=100,evaluator=eval_plugin
)


# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in scenario.train_stream:
    print("_______________________________________________________________START EXP")
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(scenario.test_stream))
    
for i in results:
    print(i["Loss_Exp/eval_phase/test_stream/Task000/Exp003"])



