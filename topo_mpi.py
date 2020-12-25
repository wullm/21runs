import py21cmfast as p21c
from math import log10, sqrt
from mpi4py import MPI

import numpy as np
from scipy import interpolate as interp
import h5py

from matplotlib import pyplot as plt

import struct; #for bytes to float
import scipy.ndimage;
from scipy import misc;
import os;

#Now, we will compute 21cm lightcones with 21cmFAST
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#The model to be run
model = "faint"

runs = 1
slices = [0]
noise_levels = np.array([0.0]);

seeds=np.array([605816,150650,278939,573177,691674,773601,727308,828958,398521,642050,310828,471108,16985,614889,40760,814979])

for run in range(runs):
	#seed=int(round(1e6*random()));
	seed = seeds[rank];
	print("Random seed", seed, rank, size);

	#Run over all the cubic slices
	for j in slices:

        #Run over all the noise levels
		for noise_lvl in noise_levels:
			#Calculate the topology
			box_fname = model + "/dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".box";
			topology_fname = model + "/topology_dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".dat";
			os.system("/cosma5/data/durham/dc-elbe1/FieldFiltrations/FieldFiltrations/triangulate " + box_fname + " > " + topology_fname);
			print("Done with topology for slice ", j, " and noise level ", noise_lvl);
