---
description: Installing Avalanche has Never Been so Simple
---

# How to Install

_Avalanche_ has been designed for extreme **portability** and **usability**. Indeed, it can be run on every OS and native python environment. 💻🍎🐧

## 📦 Install Avalanche with Pip

you can install Avalanche with pip:

```bash
pip install avalanche-lib
```

This will install the core version of Avalanche, without extra packages (e.g., object detection support, reinforcement learning support). To install all the extra packages run:

```bash
pip install avalanche-lib[all]
```

You can install also specific extra packages by specifying the appropriate code name within the square brackets. This is the complete list of options:

```bash
pip install avalanche-lib[extra] # supports for specific functionalities (e.g. specific strategies)
pip install avalanche-lib[rl] # reinforcement learning support
pip install avalanche-lib[detection] # object detection support
```

Avalanche will raise an error if you need one extra package and will suggest the appropriate package to install.

**Note** that in some alternatives to bash like zsh you may need to enclose \`avalanche-lib\[code]\` into quotation marks ( " " ), since square brackets are used as special characters.

## 📥 Install the Master Branch Using Pip

If you want, you can install Avalanche directly from the master branch (latest version) in a single command. Make sure to have **pytorch** already installed in your environment, then execute

```shell
pip install git+https://github.com/ContinualAI/avalanche.git
```

To update avalanche to the latest version, uninstall the package with `pip uninstall avalanche-lib` and then execute again the pip install command.

## 🐍 Install the Master Branch Using Anaconda

We suggest you to use the _pip package_, but if you need some recent features you may want to install directly from the **master branch**. In general, the master branch is well tested and safe to use. However, the API of new features may change more frequently or break backward compatibility. _Reproducibility is also easier if you use the pip package._

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
