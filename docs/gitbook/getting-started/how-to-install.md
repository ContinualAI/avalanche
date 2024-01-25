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

**Warning**: by installing the [all] and [extra] versions, the PyTorch version may be limited to <2.* due to the dependencies of those additional packages.

## 📥 Install the Master Branch Using Pip

If you want, you can install Avalanche directly from the master branch (latest version) in a single command. Make sure to have **pytorch** already installed in your environment, then execute

```shell
pip install git+https://github.com/ContinualAI/avalanche.git
```

To update avalanche to the latest version, uninstall the package with `pip uninstall avalanche-lib` and then execute again the pip install command.

## 💻 Developer Mode Install

To help us to expand and improve _Avalanche_, you can install Avalanche in a fresh environment with the command

```pip install -e ".[dev]"```

This will install in editable mode, so that you can develop and modify the installed Avalanche package. It will also install the "extra" dev dependencies necessary to run tests and build the documentation.

## 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory:

{% embed url="https://colab.research.google.com/drive/1pSTUgftqqg2sFNlvM6ourNYLpt2lnCQf?usp=sharing" %}
Run the "How to Install" Chapter on Google Colab
{% endembed %}
