import setuptools
import codecs
import os.path

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

setuptools.setup(
    name="avalanche-lib", # Replace with your own username
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
    python_requires='>=3.6,<3.10',
    install_requires=[
        'typing-extensions',
        'psutil',
        'gputil',
        'scikit-learn',
        'matplotlib',
        'numpy',
        'pytorchcv',
        'quadprog',
        'wandb',
        'tensorboard>=1.15',
        'pycocotools',
        'tqdm',
        'torch',
        'torchvision',
        'torchmetrics',
        'higher',
        'gdown',
        'ctrl-benchmark',
        'lvis',
        'setuptools<=59.5.0'
    ]
)
