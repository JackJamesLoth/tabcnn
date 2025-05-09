______________________________________________________________________

<div align="center">

# TabCNN PyTorch Implementation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

A simple pytorch implementation of TabCNN that I did a while back, figured it would be good to have this on GitHub.  I was not involved in the original TabCNN work at all, this was just something I threw together a year or two ago for a separate project.

[Original TabCNN repo](https://github.com/andywiggins/tab-cnn/tree/master)

[TabCNN paper](https://archives.ismir.net/ismir2019/paper/000033.pdf)

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/JackJamesLoth/tabcnn
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/
```

The packages will have to be installed but my environment is currently a mess, I will try to make a proper `requirements.txt` soon!

## How to run

Train model

```bash
python src/train.py experiment=tabcnn.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
