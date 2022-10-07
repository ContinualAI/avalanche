#!/bin/bash

################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 09-02-2021                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

python="3.9"
cuda_version="none"
help=false
accept_conda_prompts=false
use_mamba=false

while test $# -gt 0; do
         case "$1" in
              --python)
                  shift
                  python=$1
                  shift
                  ;;
              --cuda_version)
                  shift
                  cuda_version=$1
                  shift
                  ;;
              --help)
                  help=true
                  shift
                  ;;
              --yes)
                  accept_conda_prompts=true
                  shift
                  ;;
              --mamba)
                  use_mamba=true
                  shift
                  ;;
              *)
                 echo "$1 is not a recognized flag! Use --python and/or --cuda_version. Use --help to open the guide."
                 exit 1;
                 ;;
        esac
done

if [ "$help" = true ] ; then
    echo 'Usage: bash -i ./install_environment_dev.sh [OPTION]...'
    echo 'Install the avalanche development environment using conda'
    echo ''
    echo 'The scrip takes the following arguments:'
    echo ''
    echo '   --python         set the python version. Can take the values [3.7, 3.8, 3.9, 3.10], default 3.9.'
    echo '   --cuda_version   set the cuda version. You have to check the current version of cuda installed on your system and pass it as argument. If cuda is not installed or you want to use cpu pass "none". Can take the values [9.2, 10.1, 10.2, 11.0, 11.1, 11.3, 11.6, none], default none.'
    echo '   --yes            automatically answer yes to conda prompts.'
    echo '   --mamba          use mamba instead of conda.'
    echo '   --help           display this help and exit.'
    echo ''
    echo 'Examples:'
    echo '   bash -i install_environment_dev.sh --python 3.9 --cuda_version 11.6'
    echo '   bash -i install_environment_dev.sh --cuda_version none'
    exit 0
fi

conda_prompt=""
conda_executable="conda"
conda_channels="-c pytorch"
cuda_package=""

if [ "$accept_conda_prompts" = true ] ; then
    conda_prompt="-y"
fi

if [ "$use_mamba" = true ] ; then
    conda_executable="mamba"
fi

if ! [[ "$python" =~ ^(3.7|3.8|3.9|3.10)$ ]]; then
    echo "Select a python version between 3.7, 3.8, 3.9, 3.10"
    exit 1
fi

if ! [[ "$cuda_version" =~ ^(9.2|10.1|10.2|11.0|11.1|11.3|11.6|"none")$ ]]; then
    echo "Select a CUDA version between 9.2, 10.1, 10.2, 11.0, 11.1, 11.3, 11.6, none"
    exit 1
fi

if [[ "$cuda_version" = "none" ]]; then
    cuda_package="cpuonly"
    if [[ "$python" = 3.9 || "$python" = 3.10 ]]; then
        conda_channels="${conda_channels} -c=conda-forge"
    fi
else
    cuda_package="cudatoolkit=$cuda_version"
    if [[ "$python" = 3.9 || "$python" = 3.10 || "$cuda_version" = 11.1 || "$cuda_version" = 11.6 ]]; then
        conda_channels="${conda_channels} -c=conda-forge"
    fi
fi

echo "python version: $python"
echo "cuda version: $cuda_version"
echo "conda executable: $conda_executable"

set -euox pipefail
$conda_executable create -n avalanche-env python=$python -c conda-forge $conda_prompt
set +euox pipefail
source activate avalanche-env

set -euox pipefail
$conda_executable install pytorch torchvision $cuda_package $conda_channels $conda_prompt
$conda_executable env update --file environment-dev.yml
