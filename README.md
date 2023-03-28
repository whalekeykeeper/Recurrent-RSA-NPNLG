# Recurrent-RSA
This is a repo for a reproducing experiment of Bayesian pragmatics over the top of a deep neural image captioning model. 
The original code can be found at: https://github.com/reubenharry/Recurrent-RSA.

## Datasets
### MSCOCO
[Microsoft COCO Caption dataset (Chen et al., 2015)](https://arxiv.org/abs/1504.00325) can be downloaded from https://cocodataset.org/.

For our project, we would suggest downloading `2014 Train images` and/or `2014 Val images`.

### VG
[Visual Genome dataset (Krishna et al., 2017)](https://arxiv.org/abs/1602.07332) can be downloaded from http://visualgenome.org/.

For our project, we would suggest downloading the `version 1.0` of dataset. It is also what we used for evaluation.

##  Training
To train character-level models, use `train.py`. Demonstration code were included in this script.

You can choose to train with either MSCOCO or the VG dataset.

##  

## Evaluating
To run the evaluation, simply run `ev.py`.