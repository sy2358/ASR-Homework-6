import os
from general_utils import get_logger
from model import PhoneModel
from config import config
import sys
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

trainset = sys.argv[1]
assert os.path.exists(trainset), "first parameter should be training set - compiled with parse.py"

# directory for training outputs
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

print("read train data ("+trainset+")")
pkl_file = open(trainset, 'rb')
frames = pickle.load(pkl_file)
phonemes = pickle.load(pkl_file)
nphones = pickle.load(pkl_file)
idx2phn = pickle.load(pkl_file)
phn2idx = pickle.load(pkl_file)
phn2group = pickle.load(pkl_file)
print(" * ",len(frames),"sequences")
print(" * ",nphones,"phonemes")
pkl_file.close()

# dataset
print("split train into train/dev")
X_train, X_val, y_train, y_val = train_test_split(frames, phonemes, test_size=0.10, random_state=42)
train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))

# get logger
logger = get_logger(config.log_path)

# build model
print("configure phone recognition model")
model = PhoneModel(config, nphones, phn2group, idx2phn, logger=logger)
model.build()
model.train(train_data, val_data)
print("best model saved in: ",config.model_output)
