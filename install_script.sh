#!/bin/bash

################################################################################
# Copyright (c) 2021 ContinualAI Research                                      #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 09-02-2021                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: clair.continualai.org                                               #
################################################################################

python="3.8"
cuda_version="none"

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
              *)
                 echo "$1 is not a recognized flag! Use --python and/or --cuda_version."
                 exit 1;
                 ;;
        esac
done  

echo "python version : $python";
echo "cuda version : $cuda_version";

if ! [[ "$python" =~ ^(3.6|3.7|3.8)$ ]]; then
    echo "Select a python version between 3.6, 3.7, 3.8"
    exit 1
fi

if ! [[ "$cuda_version" =~ ^(9.2|10.1|10.2|11.0|"none")$ ]]; then
    echo "Select a CUDA version between 9.2 10.1, 10.2, 11.0, none"
    exit 1
fi

conda create -n avalanche-env python=$python -c conda-forge
conda activate avalanche-env
if [[ "$cuda_version" = "none" ]]; then 
    conda install pytorch torchvision cpuonly -c pytorch
else 
    conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch
fi
conda env update --file environment.yml

