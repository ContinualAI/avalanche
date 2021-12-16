---
description: Protocols and Metrics Code Examples
---

# Evaluation

{% code title=""Evaluation Pluging" Example" %}
```python
# --- CONFIG
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ---------

# --- TRANSFORMATIONS
train_transform = transforms.Compose([
    RandomCrop(28, padding=4),
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_transform = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# ---------

# --- BENCHMARK CREATION
mnist_train = MNIST('./data/mnist', train=True,
                    download=True, transform=train_transform)
mnist_test = MNIST('./data/mnist', train=False,
                   download=True, transform=test_transform)
benchmark = nc_benchmark(
    mnist_train, mnist_test, 5, task_labels=False, seed=1234)
# ---------

# MODEL CREATION
model = SimpleMLP(num_classes=benchmark.n_classes)

# DEFINE THE EVALUATION PLUGIN AND LOGGER
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics and a list of loggers.
# The evaluation plugin calls the loggers to serialize the metrics
# and save them in persistent memory or print them in the standard output.

# log to Tensorboard
tb_logger = TensorboardLogger()

# log to text file
text_logger = TextLogger(open('log.txt', 'a'))

# print to stdout
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    ExperienceForgetting(),
    StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[interactive_logger, text_logger, tb_logger])

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, test_mb_size=100,
    device=device, evaluator=eval_plugin)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience, num_workers=4)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(benchmark.test_stream, num_workers=4))
```
{% endcode %}

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% hint style="danger" %}
Notebook currently unavailable.
{% endhint %}
