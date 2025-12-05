import os
import glob
import numpy as np

def convert_npy_to bin(path, root_name)
files=glob.glob(f'{path}/{root_name}*burst#4_n*.npy')
for f in files:
    ff = np.load(f)
    ff.tofile(f'{os.path.splitext(f)[0]}.bin')
    
