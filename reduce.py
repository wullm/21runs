import numpy as np
import os
from os import listdir
from os.path import isfile, join

#Prepare filename using a standard format
def generate_fname(dir, title, model, rank, seed, slice_num, pre_str, ext):
	fname = dir + "/" + title + "_" + model + "_" + str(rank) + "_" \
			+ str(seed) + "_slice_" + str(slice_num) + "_" + pre_str + ext
	return(fname)

#The model to be run
model = "faint"
outdir = "faint"

#The slices and noise levels to be run
slices = np.arange(0,11)
noise_levels = np.array([0.0, 0.1, sqrt(0.1), 1.0, sqrt(10.0), 10.0, 1.0])
signal_levels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

#The seeds to be run
seeds=np.array([605816, 150650, 278939, 573177, 691674, 773601, 727308, 828958,
				398521, 642050, 310828, 471108, 16985, 614889, 40760, 814979])

for rank in range(16):
	seed = seeds[rank]
	for j in slices:
		for noise_lvl, signal_lvl in zip(noise_levels, signal_levels):
			#Format the signal and noise levels into a string
			nsigstr = "noise_%.1f_signal_%.1f" % (noise_lvl, signal_lvl)

			#Generate input topology filename
			topo_fname = generate_fname(outdir, "small_topology", model, rank, seed, j, nsigstr, ".dat")

			#Load the data
			X = np.loadtxt(topo_fname, skiprows=2)
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

			#Prepare the split filenames
			out_fname_0 = generate_fname(outdir, "small_topology", model, rank, seed, j, nsigstr, "_betti_0.dat")
			out_fname_1 = generate_fname(outdir, "small_topology", model, rank, seed, j, nsigstr, "_betti_1.dat")
			out_fname_2 = generate_fname(outdir, "small_topology", model, rank, seed, j, nsigstr, "_betti_2.dat")
			
			#Export the data per dimension
			np.savetxt(out_fname_0, b0)
			np.savetxt(out_fname_1, b1)
			np.savetxt(out_fname_2, b2)
