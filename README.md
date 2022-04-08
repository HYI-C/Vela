# VelaTwins
This is a collaboration between Ian Cheung and Vela Partners. All work is licensed to Vela.

## Brief
When there is a window of opportunity, smart entrepreneurs all around the world see the same opportunity. For that reason, there is almost never one company who tackles the same problem. In this project, our goal is to build a model to find similar companies when a company is given as an input.

## Data
The dataset has 655k companies with short_description, description and associated categories.

## Quickstart
# Spectroscopy data extraction algorithm

## Conda Environment


A new virtual environment, `Vela`, can be created and activated as follows.

This should be done in theory, in practice we can skip the step.

```bash
             $ cd path/to/project/root
             $ conda create -n Vela python=3
             $ conda activate Vela
(Vela) $ ...
```
## Requirements

The `requirements.txt` file specifies the dependencies 
you will need to install to use this repository. Before opening the repo, run:

```bash
(Vela) $ pip install -r requirements.txt
```

## Settings

The repo can be run with different `settings` and different. User parameters
can be defined in `custom.py`, which is extensively documented, both inline and
via the `documentation` notebook. After this step, run

```
ms = Settings()
ms.configure()
```

Please see the Sentence Similarity notebook for next steps
