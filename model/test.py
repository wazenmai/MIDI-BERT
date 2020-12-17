import os
import numpy as np
fname="/home/yh1488/remi/remi909/POP_all.npy"
exist=os.path.isfile(fname)
print("exist:",exist)
if exist:
	all=np.load(fname)
	print(all.shape)
