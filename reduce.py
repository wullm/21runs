import numpy as np
import os
from os import listdir
from os.path import isfile, join

fnames = listdir(".")

topo_data = {};
seeds = [];

#The data to be reduced
the_slice = 0
the_noise_lvl = 0

#Load the data by matching all the relevant files in this directory
for fname in fnames:
	if not isfile(fname): continue
	parts = fname.split("_")
	if (parts[0] == "topology"):
		noise_lvl = float(np.real_if_close(parts[-1][:-4]));
		seed = int(parts[4]);
		slice_ind = int(parts[6]);
		rank = int(parts[3]);
		if (the_slice == slice_ind and the_noise_lvl == noise_lvl):
			print(fname);
			topo_data[seed] = np.loadtxt(fname, skiprows=2);
			seeds.append(seed);

for seed in seeds:
	#Load the data
	X = topo_data[seed]
	#Extract the dimension column
	dim = X[:,1]
	#Load the data per dimension
	b0 = X[dim==0,2:]
	b1 = X[dim==1,2:]
	b2 = X[dim==2,2:]
	#Deselect points at the boundary
	b0 = b0[np.isinf(b0[:,1])==False]
	b1 = b1[np.isinf(b1[:,1])==False]
	b2 = b2[np.isinf(b2[:,1])==False]
	#Deselect points with zero persistence
	b0 = b0[(b0[:,1] - b0[:,0])>0]
	b1 = b1[(b1[:,1] - b1[:,0])>0]
	b2 = b2[(b2[:,1] - b2[:,0])>0]
	#Prepare the filenames
	out_fname_0 = "topology_slice_" + str(the_slice) + "_noise_" + str(the_noise_lvl) + "_b" + str(0) + "/" + str(seed) + ".dat"
	out_fname_1 = "topology_slice_" + str(the_slice) + "_noise_" + str(the_noise_lvl) + "_b" + str(1) + "/" + str(seed) + ".dat"
	out_fname_2 = "topology_slice_" + str(the_slice) + "_noise_" + str(the_noise_lvl) + "_b" + str(2) + "/" + str(seed) + ".dat"
	#Export the data per dimension
	np.savetxt(out_fname_0, b0)
	np.savetxt(out_fname_1, b1)
	np.savetxt(out_fname_2, b2)
