# parsing feat and phn file

```
python parse.py listfile name
```

will parse all the feats and phonemes file in listfile, align them, replace phonemes by their ID from phntable and dump the array of aligned feats/phoneme index in file `name.pkl`


Pickle file contains:

* feature np.array - shape (#nframes,123), float32
* phoneme idx np.array - shape (#nframe), int
* index>phone mapping
* phone>index mapping
* phone>group mapping

*Note*: phone `q` was missing in `phngroup` - new file commited in this repository

# data analysis

Numbers of file in each directory:

* `train/dr1`: 181
* `train/dr2`: 363
* `train/dr3`: 374
* `train/dr4`: 320
* `train/dr5`: 326
* `train/dr6`: 165
* `train/dr7`: 375
* `train/dr8`: 107

# how to run

## build a small train file and dev file, align and build the data files

```
head -200 feat_train.list > feat_train_small.list
python parse.py feat_train_small.list train_small
```

## train a model

* Check configuration in `config.py`.
* launch:

```
python train.py train_small.pkl
```

Output for small file above:

```
read train data (train_small.pkl)
 *  200 sequences
 *  61 phonemes
read dev data (dev.pkl)
 *  100 sequences
configure phone recognition model
Epoch 1 out of 20
10/10 [==========================>...] - ETA: 1s - train loss: 3.9356 - dev accuracy 15.78 - PER 71.33
- new best score!
Epoch 2 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 3.3228 - dev accuracy 20.44 - PER 67.05
- new best score!
Epoch 3 out of 20
10/10 [==========================>...] - ETA: 1s - train loss: 3.0407 - dev accuracy 23.82 - PER 64.05
- new best score!
Epoch 4 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.8391 - dev accuracy 25.93 - PER 62.04
- new best score!
Epoch 5 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.6935 - dev accuracy 27.68 - PER 60.23
- new best score!
Epoch 6 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.6003 - dev accuracy 29.63 - PER 58.53
- new best score!
Epoch 7 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.4916 - dev accuracy 30.41 - PER 58.68
Epoch 8 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.4084 - dev accuracy 31.52 - PER 56.79
- new best score!
Epoch 9 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.3308 - dev accuracy 32.59 - PER 56.43
- new best score!
Epoch 10 out of 20
10/10 [==========================>...] - ETA: 1s - train loss: 2.2734 - dev accuracy 34.09 - PER 55.15
- new best score!
Epoch 11 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.2211 - dev accuracy 34.71 - PER 54.25
- new best score!
Epoch 12 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.1822 - dev accuracy 35.20 - PER 54.00
- new best score!
Epoch 13 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.1416 - dev accuracy 36.14 - PER 53.28
- new best score!
Epoch 14 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.1182 - dev accuracy 36.70 - PER 52.83
- new best score!
Epoch 15 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.0896 - dev accuracy 37.28 - PER 52.46
- new best score!
Epoch 16 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.0749 - dev accuracy 37.45 - PER 52.48
Epoch 17 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.0573 - dev accuracy 37.26 - PER 52.62
Epoch 18 out of 20
10/10 [==========================>...] - ETA: 2s - train loss: 2.0353 - dev accuracy 37.27 - PER 52.53
- early stopping 3 epochs without improvement
```

## Evaluate a model

Compile the test set with:

```
python parse.py feat_test.list test
```

=> will build `test.pkl`.


Run the evaluation on `test.pkl`

```
python evaluate.py test
```


