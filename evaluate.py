import os
from general_utils import get_logger
from model import PhoneModel
from config import config
import sys
import pickle
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

testset = sys.argv[1]
assert os.path.exists(testset), "first parameter should be test set - compiled with parse.py"
# devset = sys.argv[2]
# assert os.path.exists(devset), "second parameter should be dev set"

# directory for testing outputs
if not os.path.exists(config.output_path):
    os.makedirs(config.output_path)

print("read test data ("+testset+")")
pkl_file = open(testset, 'rb')
frames = pickle.load(pkl_file)
phonemes = pickle.load(pkl_file)
nphones = pickle.load(pkl_file)
idx2phn = pickle.load(pkl_file)
phn2idx = pickle.load(pkl_file)
phn2group = pickle.load(pkl_file)
print(" * ",len(frames),"sequences")
print(" * ",nphones,"phonemes")
pkl_file.close()

test_data = list(zip(frames, phonemes))

# get logger
logger = get_logger(config.log_path)

# build model
model = PhoneModel(config, nphones, phn2group, idx2phn, logger=logger)
model.build()
model.evaluate(test_data)
