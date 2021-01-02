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

#The model to be run
model = "faint"
outdir = "faint"

#The slices and noise levels to be run
slices = np.arange(0,11)
noise_levels = np.array([0.0, 0.1, sqrt(0.1), 1.0, sqrt(10.0), 10.0, 1.0])
signal_levels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

#The seeds to be run
# seeds=np.array([605816, 150650, 278939, 573177, 691674, 773601, 727308, 828958,
# 				398521, 642050, 310828, 471108, 16985, 614889, 40760, 814979])
seeds=np.array([359948, 830947, 694085, 785293, 862157, 213451, 382758, 775384,
                414375, 229164, 200363, 763514, 790364, 25155, 540756, 357669])
seed = seeds[rank];
print("Running seed", seed, rank, size);

#Run over all the cubic slices
for j in slices:
	#Do the pure noise field
	box_fname = generate_fname(outdir, "small", model, rank, seed, j, "pure_noise", ".box")
	topology_fname = generate_fname(outdir, "small_topology", model, rank, seed, j, "pure_noise", ".dat")
	os.system("/cosma5/data/durham/dc-elbe1/FieldFiltrations/FieldFiltrations/triangulate " + box_fname + " > " + topology_fname);
	print("Done with pure noise topology", j, rank, size)
	time.sleep(1)

	#Do the noiseless pure signal field
	box_fname = generate_fname(outdir, "small", model, rank, seed, j, "noiseless", ".box")
	topology_fname = generate_fname(outdir, "small_topology", model, rank, seed, j, "noiseless", ".dat")
	os.system("/cosma5/data/durham/dc-elbe1/FieldFiltrations/FieldFiltrations/triangulate " + box_fname + " > " + topology_fname);
	print("Done with pure signal topology", j, rank, size)
	time.sleep(1)

	for noise_lvl, signal_lvl in zip(noise_levels, signal_levels):
		#Format the signal and noise levels into a string
		nsigstr = "noise_%.1f_signal_%.1f" % (noise_lvl, signal_lvl)

		#Calculate the topology
		box_fname = generate_fname(outdir, "small", model, rank, seed, j, nsigstr, ".box")
		topology_fname = generate_fname(outdir, "small_topology", model, rank, seed, j, nsigstr, ".dat")
		os.system("/cosma5/data/durham/dc-elbe1/FieldFiltrations/FieldFiltrations/triangulate " + box_fname + " > " + topology_fname);
		print("Done with topology for slice ", j, " and noise level ", noise_lvl, " and signal level", signal_lvl);
		time.sleep(1)
