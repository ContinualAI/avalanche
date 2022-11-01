import setuptools
import codecs
import os.path
from collections import defaultdict

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def get_extra_requires(path, add_all=True):
    """Currently not used. Dependencies are
    hard-coded below. We currently have a problem
    with setuptools and external extra dependency file"""
    with open(path) as fp:
        extra_deps = defaultdict(set)
        for line in fp:
            if line.strip() and not line.startswith('#'):
                tags = set()
                if ':' in line:
                    k, v = line.split(':')
                    tags.update(vv.strip() for vv in v.split(','))
                for t in tags:
                    extra_deps[t].add(k)

        # add tag `all` at the end
        if add_all:
            extra_deps['all'] = set(vv for v in extra_deps.values() for vv in v)

    return extra_deps


setuptools.setup(
    name="avalanche-lib",  # Replace with your own username
    version=get_version("avalanche/__init__.py"),
    author="ContinualAI",
    author_email="contact@continualai.org",
    description="Avalanche: a Comprehensive Framework for Continual Learning "
                "Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ContinualAI/avalanche",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7,<3.11',
    install_requires=[
        'typing-extensions',
        'psutil',
        'gputil',
        'scikit-learn',
        'matplotlib',
        'numpy',
        'pytorchcv',
        'wandb',
        'tensorboard>=1.15',
        'tqdm',
        'torch',
        'torchvision',
        'torchmetrics',
        'gdown',
        'quadprog',
        'dill',
        'setuptools<=59.5.0'
    ],
    extras_require=get_extra_requires('extra_dependencies.txt',
                                      add_all=True),
    include_package_data=True
)

