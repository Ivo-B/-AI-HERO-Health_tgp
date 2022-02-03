<div align="center">

# AI-HERO-Health tgp

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

What it does

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/Ivo-B/AI-HERO-Health_tgp
cd AI-HERO-Health_tgp

# [OPTIONAL] create virtual environment
python -m venv health_env
source health_env/bin/activate
pip install -U pip

pip install cython
pip install numpy
# install requirements
pip install -r requirements.txt
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=experiment_name.yaml

# train on CPU
python run.py trainer.gpus=0 experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

## Test the final Evaluation

The final ranking depends on the consumed energy for development as well as running inference and additionally on the Accuracy on the test set.
You can have a look at the calculation of the ranks in `ranking.py`, however, it is not needed in your final repository.
In order to allow us to run your model on the test set, you need to adapt the evaluation files from this repository.
The most important one is `run_eval.py`. It will load your model, run inference and save the predictions as a csv.
It is in your responsibility that this file loads the model and weights you intend to submit. 
You can test the script by just running it, it will automatically predict on the validation set during the development.
In the folder `evaluation` you find the corresponding bash scripts `eval.sh` or `eval_conda.sh` to be able to test the evaluation on HAICORE like this:
    
    cd /hkfs/work/workspace/scratch/im9193-H5/AI-HERO-Health
    sbatch evaluation/eval.sh

In the bash scripts you again need to adapt the paths to your workspace and also insert the correct model weights path.

After the csv is created, the Accuracy is calculated using `calc_score.py`, which again will use the validation set during development.
You do not need to adapt this file, for evaluating on the test set the organizers will use their own copy of this file.
Nevertheless you can test if your created csv works by running the appropriate bash script:

     sbatch evaluation/score_group.sh

For that you need to adapt the group workspace in lines 9 and 11.

For calculating the groups' final scores the mentioned files need to work. That means, that your workspace needs to contain the virtual environment that is loaded, the code as well as model weights.
To make the submission FAIR you additionally have to provide your code on Github (with a requirements file that reproduces your full environment), and your weights uploaded to Zenodo.
You can complete your submission here: https://ai-hero-hackathon.de/.
We will verify your results by also downloading everything from Github and Zenodo to a clean new workspace and check whether the results match.
