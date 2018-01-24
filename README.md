# Fake News 

My implementation of the model developed by [UCLA](https://github.com/uclmr/fakenewschallenge/)
for the [Fake News Challenge](http://www.fakenewschallenge.org). The paper describing the model
can be found [here](https://arxiv.org/pdf/1707.03264.pdf).

## How to run

Initially, ensure that you have the necessary dependencies by running `pip install -r requirements`.
Notice that this code was developed for `python>=3.5`. `cloudpickle` is used instead of vanilla
`pickle` due to issues on this latter package when storing objects larger than 2Gb.

First you need to extract the features from the train and test data. This can be done using the code
on `preprocessing.ipynb`. After running it, two pickle files will be generated, namely:
`dataset_test_encoded.pyk` and `dataset_train_encoded.pyk`. This allows running the preprocessing
phase only once, saving a considerable amount of time when iterativelly tunning the model.

Following, the model can be trained with `python trainer.py`. The model use the same parameters from the
original implementation, except for the amount of epochs, which was reduced from 90 to 30, while keeping
mostly the same results while decreasing the amount of iterations (and time) by 1/3.

To predict on the test set, use `python predictor.py`. which will load the trained model, the
test samples and generate a file named `predicted_test.csv`. Finally it's possible to check
the performance of the model with `python scorer.py gold_labels.csv predicted_tests.csv`. This script
is simply a copy from the Fake News Challenge repository and provides the model score, accuracy and
confusion matrix.

## Reported results

The obtained results are reproduced below:

```
CONFUSION MATRIX:
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |   1110    |     3     |    666    |    124    |
-------------------------------------------------------------
| disagree  |    308    |    26     |    234    |    129    |
-------------------------------------------------------------
|  discuss  |    845    |    20     |   3270    |    329    |
-------------------------------------------------------------
| unrelated |    95     |     0     |    289    |   17965   |
-------------------------------------------------------------
ACCURACY: 0.880

MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||
|| 11651.25  ||  4587.25  ||  9416.25  ||
```
