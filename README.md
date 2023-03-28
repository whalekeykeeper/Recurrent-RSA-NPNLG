# Recurrent-RSA
This is a repo for a reproducing experiment of Bayesian pragmatics over the top of a deep neural image captioning model. 
The original code can be found at: https://github.com/reubenharry/Recurrent-RSA

## Setting up evaluation

Run the following command in the appropriate environment to generate the test sets for evaluation of the models:

```bash
python build_test_sets.py
```

After the test sets have been generated, the following command runs the actual evaluation script:

```bash
python evaluation.py
```