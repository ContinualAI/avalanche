---
description: Installing Avalanche has Never Been so Simple
---

# How to Install

_Avalanche_ has been designed for extreme **portability** and **usability**. Indeed, it can be run on every OS and native python environment. 💻🍎🐧

In order to install _Avalanche_ we have three main options:

1. [Installing it with Pip](how-to-install.md#installing-avalanche-with-pip)
2. [Installing it with Anaconda](how-to-install.md#install-avalanche-with-anaconda)
3. [Developer Mode Install](how-to-install.md#developer-mode-install) \(for contributing to _Avalanche_!\)

## 🔂 Avalanche Dependencies

The _Avalanche_ dependencies are the following:

`python>=3.6,<=3.9.2`, `typing-extensions`, `psutil`, `torch`, `torchvision`, `tensorboard`, `scikit-learn`, `matplotlib`, `numpy`, `pytorchcv`, `quadprog`, `tqdm`, `googledrivedownloader`.

{% hint style="info" %}
_Avalanche may work on lower Python versions as well but we don't officially support it, nor recommend it._
{% endhint %}

At the moment, we cannot provide a swift installation experience as some of the dependencies cannot be installed automatically. However, in the sections below we detail how you can install Avalanche **in a matter of minutes** on any platform!

## 📦 Installing Avalanche with Pip

Within an Anaconda environment or not you can install _Avalanche_ also with Pip with the following steps:

1. Install the package with pip.
2. [Install Pytorch + TorchVision](https://pytorch.org/) \(follow the instructions on the website to use pip\)

Step 1. can be done with the following line of code:

```bash
pip install git+https://github.com/ContinualAI/avalanche.git
```

That's it. now we have _Avalanche_ up and running and we can start using it!

## 🐍 Install Avalanche with Anaconda

This is the **safest option** since it allows you to build an isolated environment for your Avalanche experiments. This means that you'll be able to _pin particular versions_ of your dependencies that may differ from the ones you want to maintain in the rest of your system. This will in turn increase reproducibility of any experiment you may produce.

Assuming you have **Anaconda \(or Miniconda\) installed** on your system, you can follow these simple steps:

1. Install the `avalanche.yml` environment and activate it.
2. [Install Pytorch + TorchVision](https://pytorch.org/) \(follow the instructions on the website to use conda\)
3. Update the Conda Environment

These steps can be accomplished with the following lines of code:

```bash
# choose you python version
python="3.8"

# Step 1
git clone https://github.com/vlomonaco/avalanche.git
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

If you want to expand _Avalanche_ and help us improve it \(see the "[_From Zero to Hero_](../from-zero-to-hero-tutorial/2.-benchmarks.md)" Tutorial\). In this case we suggest to create an environment in _**developer-mode**_ as follows \(just a couple of more dependencies will be installed\).

Assuming you have **Anaconda \(or Miniconda\) installed** on your system, you can follow these simple steps:

1. Install the `avalanche-dev.yml` environment and activate it.
2. [Install Pytorch + TorchVision](https://pytorch.org/) \(follow the instructions on the website to use conda\).
3. Update the Conda Environment.

These three steps can be accomplished with the following lines of code:

```bash
# choose you python version
python="3.8"

# Step 1
git clone https://github.com/vlomonaco/avalanche.git
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

{% embed url="https://colab.research.google.com/drive/1pSTUgftqqg2sFNlvM6ourNYLpt2lnCQf?usp=sharing" caption="Run the \"How to Install\" Chapter on Google Colab" %}

