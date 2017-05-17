# read feat and phn files

import struct
import numpy as np

# read feat file, and returns, nframe, sample_period, sample_size, parameters, and numpy array shaped (nframe, sample_size)
def parse_feat(file):
  with open(file, "rb") as f:
    nframe, = struct.unpack('>i4',f.read(4))
    sample_period, = struct.unpack('>i4',f.read(4))
    sample_size, = struct.unpack('>h2',f.read(2))
    parameters, = struct.unpack('>h2',f.read(2))
    data = np.fromstring(f.read(nframe*sample_size), dtype='>f4')
    return nframe, sample_period, sample_size/4, parameters, np.reshape(data, (nframe,-1))

# for each phn file, returns an array [begin, end, phoneme]
def parse_phn(file):
  phn_desc = []
  with open(file, "r") as f:
    for line in f:
      line = line.strip()
      l = line.split(' ')
      phn_desc.append([l[0], l[1], l[2]])
  return phn_desc

if __name__ == "__main__":
    import sys
    nframe, sample_period, sample_size, parameters, data = parse_feat(sys.argv[1]+".feat")
    print '#frames:', nframe
    print 'sample period:', sample_period
    print 'sample size (x4):', sample_size
    print 'parameters:', parameters
    print 'feats:', data.shape
    phn_sequence = parse_phn(sys.argv[1]+".phn")
    print 'phn sequence:', phn_sequence
