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

python="3.8"
cuda_version="none"
help=false

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
              *)
                 echo "$1 is not a recognized flag! Use --python and/or --cuda_version. Use --help to open the guide."
                 exit 1;
                 ;;
        esac
done

if [ "$help" = true ] ; then
    echo 'Usage: bash -i ./install_environment.sh [OPTION]...'
    echo 'Install the avalanche environment using conda'
    echo ''
    echo 'The scrip takes the following arguments:'
    echo ''
    echo '   --python         set the python version. Can take the values [3.6, 3.7, 3.8, 3.9], default 3.8.'
    echo '   --cuda_version   set the cuda version. You have to check the current version of cuda installed on your system and pass it as argument. If cuda is not installed or you want to use cpu pass "none". Can take the values [9.2, 10.1, 10.2, 11.0, 11.1, none], default none.'
    echo '   --help           display this help and exit.'
    echo ''
    echo 'Examples:'
    echo '   bash -i install_environment.sh --python 3.7 --cuda_version 10.2'
    echo '   bash -i install_environment.sh --cuda_version none'
    exit 0
fi

echo "python version : $python";
echo "cuda version : $cuda_version";

if ! [[ "$python" =~ ^(3.6|3.7|3.8|3.9)$ ]]; then
    echo "Select a python version between 3.6, 3.7, 3.8, 3.9"
    exit 1
fi

if ! [[ "$cuda_version" =~ ^(9.2|10.1|10.2|11.0|11.1|"none")$ ]]; then
    echo "Select a CUDA version between 9.2 10.1, 10.2, 11.0, 11.1, none"
    exit 1
fi

conda create -n avalanche-env python=$python -c conda-forge
conda activate avalanche-env
if [[ "$cuda_version" = "none" ]]; then
    if [[ "$python_version" = 3.9 ]]; then
        conda install pytorch torchvision cpuonly -c pytorch -c=conda-forge
    else
        conda install pytorch torchvision cpuonly -c pytorch
    fi
else
    if [[ "$python_version" = 3.9 || "$cuda_version" = 11.1 ]]; then
        conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch -c=conda-forge
    else
        conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch
    fi
fi
conda env update --file environment.yml
