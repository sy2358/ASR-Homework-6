# parsing feat and phn file

```
python parse.py listfile name
```

will parse all the feats and phonemes file in listfile, align them, replace phonemes by their ID from phntable and dump the array of aligned feats/phoneme index in file `name.pkl`

For each file, features are in a (#nframes,123) float numpy array, and phonemes are in a (#nframes) int numpy array.

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
tail -100 feat_train.list > feat_dev.list
python parse.py feat_dev.list dev
```

## train a model

* Check configuration in `config.py`.
* launch:

```
python train.py train_small.pkl dev.pkl
```

# Details

The model is:

* a bidirectional LSTM built using `tf.nn.bidirectional_dynamic_rnn` - input is the sequence of frames
* dropout layer for the training
* a logit layer - generating nphones label

