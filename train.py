import os
from general_utils import get_logger
from model import PhoneModel
from config import config
import sys
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

trainset = sys.argv[1]
assert os.path.exists(trainset), "first parameter should be training set"
devset = sys.argv[2]
assert os.path.exists(devset), "second parameter should be dev set"

# directory for training outputs
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

print("read train data ("+trainset+")")
pkl_file = open(trainset, 'rb')
train_data = pickle.load(pkl_file)
nphones = pickle.load(pkl_file)
idx2phn = pickle.load(pkl_file)
phn2idx = pickle.load(pkl_file)
phn2group = pickle.load(pkl_file)
print(" * ",len(train_data),"sequences")
print(" * ",nphones,"phonemes")
pkl_file.close()

print("read dev data ("+devset+")")
pkl_file = open(devset, 'rb')
dev_data = pickle.load(pkl_file)
print(" * ",len(dev_data),"sequences")
pkl_file.close()

# get logger
logger = get_logger(config.log_path)

# build model
print("configure phone recognition model")
model = PhoneModel(config, nphones, phn2group, idx2phn, logger=logger)
model.build()

model.train(train_data, dev_data)

print("best model saved in: ",config.model_output)
