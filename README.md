# Recurrent-RSA
This is a repo for a reproducing experiment of Bayesian pragmatics over the top of a deep neural image captioning model. 
The original code can be found at: https://github.com/reubenharry/Recurrent-RSA.

## Datasets
### MSCOCO
[Microsoft COCO Caption dataset (Chen et al., 2015)](https://arxiv.org/abs/1504.00325) can be downloaded from https://cocodataset.org/.

For our project, we suggest downloading `2014 Train images` and/or `2014 Val images`.

### VG
[Visual Genome dataset (Krishna et al., 2017)](https://arxiv.org/abs/1602.07332) can be downloaded from http://visualgenome.org/.

For our project, we suggest downloading the `version 1.0` of dataset. It is also what we used for evaluation.

##  Training
To train an image captioning model, use `train.py`. Demonstration code is included in this script and can be adjusted to use different datasets/hyperparameters.

You can choose to train with either MSCOCO or the VG dataset.

##  

## Setting and running up evaluation

The datasets are expected to be put into the `data/` folder (but paths can be adjusted in the code).
Run the following command in the appropriate environment to generate the test sets for evaluation of the models:

```bash
python build_test_sets.py
```

After the test sets have been generated, the following command runs the actual evaluation script:

```bash
python evaluation.py
```

##

## Project Layout

- `bayesian_agents/` contains the `RSA` model
- `data/` contains datasets (need to be downloaded and put there manually due to their file size)
- `evaluation/` contains the evaluation module
- `train/` contains the image captioning model used by the `RSA`
- `recursion_schemes/` contains greedy/beam search
- `utils/` contains various utility functions mainly took over from the original codebase
- `paper` contains the original paper
