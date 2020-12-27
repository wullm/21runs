from mpi4py import MPI
from math import sqrt
import numpy as np
import os;
import time

#Prepare filename using a standard format
def generate_fname(dir, title, model, rank, seed, slice_num, pre_str, ext):
	fname = dir + "/" + title + "_" + model + "_" + str(rank) + "_" \
			+ str(seed) + "_slice_" + str(slice_num) + "_" + pre_str + ext
	return(fname)

#Size of the cluster
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#The first model of the pair
model1 = "faint"
outdir1 = "faint"

#The second model of the pair
model2 = "faint"
outdir2 = "faint"

#Are the two models identical?
identical = (model1 == model2) and (outdir1 == outdir2)

#The slices and noise levels to be run
slices = np.array([0])
noise_levels = np.array([0.0])
signal_levels = np.array([1.0])

#The seeds to be run
seeds=np.array([605816, 150650, 278939, 573177, 691674, 773601, 727308, 828958,
				398521, 642050, 310828, 471108, 16985, 614889, 40760, 814979])
ranks = range(len(seeds))

#Make a list of all unique pairs
pairs = []
for rank1 in ranks:
	for rank2 in ranks:
		if (rank1 == rank2): continue
		if ((rank2, rank1) in pairs and identical): continue
		pairs.append((rank1, rank2))

pairs = np.array(pairs)

#Each rank will output distances to a separate file
outfile = "topo_distances_" + outdir1 + "_" + outdir2 + "_" + str(rank) + ".dat"

#Run over all pairs assigned to this rank
for pair in pairs[rank::size]:
	rank1 = pair[0]
	rank2 = pair[1]
	seed1 = seeds[rank1]
	seed2 = seeds[rank2]

	print("Doing pair", seed1, seed2)

	#Run over all the cubic slices
	for j in slices:

		for noise_lvl, signal_lvl in zip(noise_levels, signal_levels):
			#Format the signal and noise levels into a string
			nsigstr = "noise_%.1f_signal_%.1f" % (noise_lvl, signal_lvl)

			#Calculate the topological distance for each dimension
			for dim in range(3):
				topo_fname1 = generate_fname(outdir1, "small_topology", model1, rank1, seed1, j, nsigstr, "_betti_" + str(dim) +".dat")
				topo_fname2 = generate_fname(outdir2, "small_topology", model2, rank2, seed2, j, nsigstr, "_betti_" + str(dim) +".dat")
				os.system("echo -n " seed1 + " " + seed2 + " " + j + " " + noise_lvl + " " + signal_lvl + " " + dim + " >> " + outfile)
				os.system("/cosma5/data/durham/dc-elbe1/hera/hera/geom_matching/wasserstein/wasserstein_dist " + topo_fname1 + " " + topo_fname2 " >> " + outfile);
				time.sleep(0.5)
