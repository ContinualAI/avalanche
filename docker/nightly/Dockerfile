# syntax=docker/dockerfile:1

FROM continualai/avalanche-test-3.9:latest

# Needed for quadprog
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN echo "conda activate avalanche-env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pip install git+https://github.com/ContinualAI/avalanche.git
