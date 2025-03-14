# Contribute to Avalanche

### ]--- description: How to Contribute Back to the Avalanche Community

## Contribute to Avalanche

The last step towards becoming a **real continual learning super-hero** ⚡ is to fall into a _radioactive dump_.☢️ Just kidding, it's much easier than that: you need to contribute back to _Avalanche_!

There are no _superheroes_ that are not altruistic!

{% hint style="info" %}
In order to contribute to _Avalanche_, first of all you need to become familiar with all its features and the codebase structure, so if you have not followed the _"From Zero to Hero Tutorial"_ from the beginning we suggest to do it before starting to make changes.
{% endhint %}

First of all, [install _Avalanche_ in "_Developer Mode"_](https://avalanche.continualai.org/getting-started/how-to-install#developer-mode-install) _if you haven't already._ After you've familiarized with the _Avalanche_ codebase you have two roads ahead of you:

1. You can start working on a [open issue](https://avalanche.continualai.org/questions-and-issues/add-your-issue) (we have dozen of them!)
2. You can [submit a feature-request](https://avalanche.continualai.org/questions-and-issues/request-a-feature) and propose yourself to work on it.

![Examples of Avalanche Issues available on GitHub](../../.gitbook/assets/issues.png)

**In any of the two cases you'd need to follow the steps below**:

1. [Join our Slack](https://join.slack.com/t/continualai/shared_invite/enQtNjQxNDYwMzkxNzk0LTBhYjg2MjM0YTM2OWRkNDYzOGE0ZTIzNDQ0ZGMzNDE3ZGUxNTZmNmM1YzJiYzgwMTkyZDQxYTlkMTI3NzZkNjU) and **#avalanche-dev** channel (optional but recommended)
2. ⭐\_Star\_ + 👁️\_watch\_ the repository.
3. Fork the repository.
4. Create or assign an existing issue/feature to yourself.
5. Make your changes.
6. Make a [Pull Request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-requests) (PR).

**The following rules should be respected:**

* Use [_Black_](https://github.com/psf/black) code formatting for a consistent coding style, which also handles line lengths (the 88 columns limit) automatically.
* Always pull before pushing a commit.
* Try to assign to yourself one issue at a time.
* Try closing an issue **within roughly 7 days**. If you are not able to do that, please break it down into multiple ones you can tackle more easily, or you can always remove your assignment to the issue!
* If you add a new feature, please include also a _test_ and a usage example in your PR.

Also, before making your PR, first run the following command for code formatting with _Black_:

```bash
black .
```

then, **make sure** that the following command returns no test errors:

```bash
python -m unittest discover tests -v
```

Otherwise fix them and run these commands again until everything is working correctly. You should also check if everything is working on GPUs, using the env variable `USE_GPU=True`:

```bash
USE_GPU=True python -m unittest discover tests -v
```

Faster integrity checks can be run with the env variable `FAST_TEST=True` :

```bash
USE_GPU=False FAST_TEST=True python -m unittest discover tests -v
```

**Contribute to the Avalanche documentation**

Apart from the code, you can also contribute to the Avalanche documentation 📚! We use [Jupyter notebooks](https://jupyter.org/) to write the documentation, so both code and text can be smoothly inserted, and, as you may have noticed, all our documentation can be run on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)!

To contribute to the documentation you need to follow the steps below:

1. The notebooks are contained in the folder `notebooks`. The folder structure is specular to the documentation, so do not create or delete any folder.
2. Detect the notebook that you want to edit and do all the modifications 📝
3. Commit the changes and open a pull request (PR).
4. If your pull request will be accepted, your edited notebooks will be automatically converted and uploaded to the official Avalanche website 🎊!

### 🤝 Run it on Google Colab

You can run _this chapter_ and play with it on Google Colaboratory: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ContinualAI/avalanche/blob/master/notebooks/from-zero-to-hero-tutorial/09_contribute-to-avalanche.ipynb)
