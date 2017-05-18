# read feat and phn files

import struct
import numpy as np
import math
import sys
import pickle

# read phntable and build maps idx2phn and phn2idx
# return also the number of phones
def read_phntable(file):
  idx = 0
  idx2phn = {}
  phn2idx = {}
  with open(file, "r") as f:
    for line in f:
      line = line.strip()
      phn2idx[line] = idx
      idx2phn[idx] = line
      idx = idx + 1
  return idx2phn, phn2idx, idx

# read the phoneme table
idx2phn, phn2idx, nphones = read_phntable('phntable')

# read feat file, and returns, nframe, sample_period, sample_size, parameters, and numpy array shaped (nframe, sample_size)
def parse_feat(file):
  with open(file, "rb") as f:
    nframe, = struct.unpack('>i',f.read(4))
    sample_period, = struct.unpack('>i',f.read(4))
    sample_size, = struct.unpack('>h',f.read(2))
    parameters, = struct.unpack('>h',f.read(2))
    data = np.fromstring(f.read(nframe*sample_size), dtype='>f')
    return nframe, sample_period, sample_size/4, parameters, np.reshape(data, (nframe,-1))

# for each phn file, returns an array [begin, end, phoneme]
def parse_phn(file):
  phn_desc = []
  with open(file, "r") as f:
    for line in f:
      line = line.strip()
      l = line.split(' ')
      phn_desc.append([int(l[0]), int(l[1]), l[2]])
  return phn_desc

# align feat (bidimensional np array) and phonemes - convert phn_sequence in an array of idx - same height than feats
def alignFeatPhonem(phn2idx, feats, phn_sequence):
  phn = np.zeros((feats.shape[0]), dtype=np.int)
  start = 0
  end = 0
  idx = 0
  for seq in phn_sequence:
    idx = phn2idx[seq[2]]
    start = int(math.floor(seq[0]/0.01/16000))
    end = min(int(math.floor(seq[1]/0.01/16000)),feats.shape[0])
    for h in range(start, end):
      phn[h] = idx
  # complete if not going to end of the frames
  for h in range(end, feats.shape[0]):
    phn[h] = idx
  return phn

filelist = sys.argv[1]
setname = sys.argv[2]

# check we have provided filelist and name for the set
assert(setname and len(setname)>0)

data = []
with open(filelist, "r") as fl:
  for line in fl:
    line = line.strip()
    nframe, sample_period, sample_size, parameters, feats = parse_feat(line)
    phn_sequence = parse_phn(line.replace(".feat",".phn"))
    phn = alignFeatPhonem(phn2idx, feats, phn_sequence)
    print("read "+line+" - nframes=",nframe)
    data.append((feats,phn))

print("read "+str(len(data))+" files - dump into "+setname+'.pkl')
output = open(setname+'.pkl', 'wb')
pickle.dump(data, output)
pickle.dump(nphones, output)
output.close()
