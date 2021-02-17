---
description: Benchmarks and DatasetCode Examples
---

# Benchmarks

{% code title="\"All MNIST\" Example" %}
```python
# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model = SimpleMLP(num_classes=10)

# Here we show all the MNIST variation we offer in the "classic" benchmarks
# scenario = PermutedMNIST(n_steps=5, seed=1)
scenario = RotatedMNIST(n_steps=5, rotations_list=[30, 60, 90, 120, 150], seed=1)
# scenario = SplitMNIST(n_steps=5, seed=1)

# choose some metrics and evaluation method
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, task=True),
    loss_metrics(minibatch=True, epoch=True, task=True),
    timing_metrics(epoch=True, epoch_average=True, test=False),
    cpu_usage_metrics(step=True),
    Forgetting(),
    loggers=[interactive_logger])

 # Than we can extract the parallel train and test streams
train_stream = scenario.train_stream
test_stream = scenario.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2,
    test_mb_size=32, device=device, evaluator=eval_plugin
)

# train and test loop
results = []
for train_task in train_stream:
    print("Current Classes: ", train_task.classes_in_this_step)
    cl_strategy.train(train_task, num_workers=4)
    mresults.append(cl_strategy.test(test_stream))
```
{% endcode %}

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% hint style="danger" %}
Notebook currently unavailable.
{% endhint %}

