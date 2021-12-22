---
description: Installing Avalanche has Never Been so Simple
---

# How to Install

_Avalanche_ has been designed for extreme **portability** and **usability**. Indeed, it can be run on every OS and native python environment. 💻🍎🐧

## 📦 Installing Avalanche with Pip

you can install Avalanche with pip:

```bash
pip install avalanche-lib
```

That's it. Now you can start using Avalanche.

## Installing the Master Branch Using Anaconda

We suggest you to use the pip package, but if you need some recent features you may want to install directly from the master branch. In general, the master branch is well tested and safe to use. However, the API of new features may change more frequently or break backward compatibility. Reproducibility is also easier if you use the pip package.

```bash
# choose your python version
python="3.8"

# Step 1
git clone https://github.com/ContinualAI/avalanche.git
cd avalanche
conda create -n avalanche-env python=$python -c conda-forge
conda activate avalanche-env

# Step 2
# Istall Pytorch with Conda (instructions here: https://pytorch.org/)

# Step 3
conda env update --file environment.yml
```

{% hint style="info" %}
On **Linux**, alternatively, you can simply run the `install_environment.sh` in the _Avalanche_ home directory. The script takes 2 arguments: `--python` and `--cuda_version`. Check `--help` for details.
{% endhint %}

You can test your installation by running the `examples/test_install.py` script. Make sure to include avalanche into your **$PYTHONPATH** if you are running examples with the command line interface.

## 💻 Developer Mode Install

If you want to expand _Avalanche_ and help us improve it (see the "[_From Zero to Hero_](../from-zero-to-hero-tutorial/03\_benchmarks.md)" Tutorial). In this case we suggest to create an environment in _**developer-mode**_ as follows (just a couple of more dependencies will be installed).

Assuming you have **Anaconda (or Miniconda) installed** on your system, you can follow these simple steps:

1. Install the `avalanche-dev-env` environment and activate it.
2. [Install Pytorch + TorchVision](https://pytorch.org) (follow the instructions on the website to use conda).
3. Update the Conda Environment.

These three steps can be accomplished with the following lines of code:

```bash
# choose you python version
python="3.8"

# Step 1
git clone https://github.com/ContinualAI/avalanche.git
cd avalanche
conda create -n avalanche-dev-env python=$python -c conda-forge
conda activate avalanche-dev-env

# Step 2
# Istall Pytorch with Conda (instructions here: https://pytorch.org/)

# Step 3
conda env update --file environment-dev.yml
```

{% hint style="info" %}
On **Linux**, alternatively, you can simply run the `install_environment_dev.sh` in the _Avalanche_ home directory. The script takes 2 arguments: `--python` and `--cuda_version`. Check `--help` for details.
{% endhint %}

You can test your installation by running the `examples/test_install.py` script. Make sure to include avalanche into your **$PYTHONPATH** if you are running examples with the command line interface.

That's it. now we have _Avalanche_ up and running and we can start contribute to it!

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% embed url="https://colab.research.google.com/drive/1pSTUgftqqg2sFNlvM6ourNYLpt2lnCQf?usp=sharing" %}
Run the "How to Install" Chapter on Google Colab
{% endembed %}
