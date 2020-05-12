# OpenLORIS-Object benchmarks
We are testing the performances based on OpenLORIS-Object dataset

This is the implementation of the following paper: 
Qi She et al, 
[OpenLORIS-Object: A Robotic Vision Dataset and Benchmark for Lifelong Deep Learning](https://arxiv.org/pdf/1911.06487.pdf)
The paper has been accepted into ICRA 2020. 

## Requirements
The current version of the code has been tested with following libs:
* `pytorch 1.1.0`
* `torchvision 0.2.1`
* `tqdm`
* `visdom`
* `Pillow`

Install the required the packages inside the virtual environment:
```
$ conda create -n yourenvname python=3.7 anaconda
$ source activate yourenvname
$ pip install -r requirements.txt
```

## Data Preparation
Step 1: Download data following [Google Drive](https://drive.google.com/open?id=1KlgjTIsMD5QRjmJhLxK4tSHIr0wo9U6XI5PuF8JDJCo). 

Step 2: Run following scripts:
```
 python3 benchmark1.py
 python3 benchmark2.py
```

Step 3: Put train/test/validation file under `./img`. For more details, please follow `note` file under each sub-directories in `./img`.

Step 4: Generate the `.pkl` files of data.
```
 python3 pk_gene.py
 python3 pk_gene_sequence.py
```
## Quickly get hands on

You can directly use scripts on 9 algorithms and 2 benchmarks (may need to modify arguments/parameters in `.bash` files if necessary):
```
bash clutter.bash
bash illumination.bash
bash pixel.bash
bash occlusion.bash
bash sequence.bash
```

## Running Benchmark 1
Individual experiments can be run with `main.py`. Main option is:

```
python3 main.py --factor
```

which kind of experiment? (`clutter`|`illumination`|`occlusion`|`pixel`)


## Running Benchmark 2
The main option to run benchmark2 is:

```
python3 main.py --factor=sequence
```

## Running specific baseline methods

- Elastic weight consolidation (EWC): 

```
main.py --ewc --savepath=ewc
```
- Online EWC:  

```
main.py --ewc --online --savepath=ewconline
```

- Synaptic intelligence (SI): 

```
main.py --si --savepath=si
```
- Learning without Forgetting (LwF): 

```
main.py --replay=current --distill --savepath=lwf
```

- Deep Generative Replay (DGR): 

```
main.py --replay=generative --savepath=dgr
```

- DGR with distillation: 

```
main.py --replay=generative --distill --savepath=distilldgr
```

- Replay-trough-Feedback (RtF): 

```
main.py --replay=generative --distill --feedback --savepath=rtf
```

- Cumulative: 

```
main.py --cumulative=1 --savepath=cumulative
```

- Naive: 

```
main.py --savepath=naive
```


## Repository Structure
```
OpenLORISCode 
├── img
├── lib
│   ├── callbacks.py
│   ├── continual_learner.py
│   ├── encoder.py
│   ├── exemplars.py
│   ├── replayer.py
│   ├── train.py
│   ├── vae_models.py
│   ├── visual_plt.py
├── _compare.py
├── _compare_replay.py
├── _compare_taskID.py
├── data.py
├── evaluate.py
├── excitability_modules.py
├── main.py
├── linear_nets.py
├── param_stamp.py
├── pk_gene.py
├── visual_visdom.py
├── utils.py
└── README.md
```
## Citation 
Please consider citing our papers if you use this code in your research:
```
@misc{1911.06487,
  Author = {Qi She and Fan Feng and Xinyue Hao and Qihan Yang and Chuanlin Lan and Vincenzo Lomonaco and Xuesong Shi and Zhengwei Wang and Yao Guo and Yimin Zhang and Fei Qiao and Rosa H. M. Chan},
  Title = {OpenLORIS-Object: A Dataset and Benchmark towards Lifelong Object Recognition},
  Year = {2019},
  Eprint = {arXiv:1911.06487},
}
```

## Acknowledgements
Parts of code were borrowed from [here](https://github.com/GMvandeVen/continual-learning).


## Issue / Want to Contribute ? 
Open a new issue or do a pull request in case you are facing any difficulty with the code base or if you want to contribute.




