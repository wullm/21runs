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

#Open a topology file and split into separate files for each dimension
def split_file(fname):
	#Load the data
	X = np.loadtxt(fname, skiprows=2)
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
	out_fname_0 = fname + ".0"
	out_fname_1 = fname + ".1"
	out_fname_2 = fname + ".2"

	#Export the data per dimension
	np.savetxt(out_fname_0, b0)
	np.savetxt(out_fname_1, b1)
	np.savetxt(out_fname_2, b2)


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
slices = np.arange(0,11)
noise_levels = np.array([0.0, 0.1, sqrt(0.1), 1.0, sqrt(10.0), 10.0, 1.0])
signal_levels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

#The seeds to be run
seeds1=np.array([605816, 150650, 278939, 573177, 691674, 773601, 727308, 828958,
				 398521, 642050, 310828, 471108, 16985, 614889, 40760, 814979])
seeds2=np.array([359948, 830947, 694085, 785293, 862157, 213451, 382758, 775384,
                 414375, 229164, 200363, 763514, 790364, 25155, 540756, 357669])
seeds = np.append(seeds1,seeds2)
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
outfile = "distances/distances_" + outdir1 + "_" + outdir2 + "_" + str(rank) + ".dat"

#Run over all pairs assigned to this rank
for pair in pairs[rank::size]:
	rank1 = pair[0]
	rank2 = pair[1]
	seed1 = seeds[rank1]
	seed2 = seeds[rank2]

	print("Doing pair", seed1, seed2)

	#We will need to unpack the relevant files from the tars
	tar_name1 = outdir1 + "_topo.tar.gz"
	tar_name2 = outdir2 + "_topo.tar.gz"
	for j in slices:
		for noise_lvl, signal_lvl in zip(noise_levels, signal_levels):
			#Format the signal and noise levels into a string
			nsigstr = "noise_%.1f_signal_%.1f" % (noise_lvl, signal_lvl)
			nsig_only = "%.1f %.1f" % (noise_lvl, signal_lvl)

			#Format the corresponding filenames
			topo_fname1 = generate_fname(outdir1, "small_topology", model1, rank1 % 16, seed1, j, nsigstr, ".dat")
			topo_fname2 = generate_fname(outdir2, "small_topology", model2, rank2 % 16, seed2, j, nsigstr, ".dat")
			#Extract those files
			os.system("tar -xvf " + tar_name1 + " " + topo_fname1)
			os.system("tar -xvf " + tar_name2 + " " + topo_fname2)

			#Create separate files for each dimension
			split_file(topo_fname1)
			split_file(topo_fname2)

			#Calculate the topological distance for each dimension
			for dim in range(3):
				#The files containing the persistence diagrams for this dimension
				fname1 = topo_fname1 + "." + str(dim)
				fname2 = topo_fname2 + "." + str(dim)
				#Output data for the other columns
				cols = "topo " + str(seed1) + " " + str(seed2) + " " + str(j) + " " + nsig_only + " " + str(dim) + " "
				os.system("echo -n \"" + cols + "\">> " + outfile)
				#Compute the distance and output it
				os.system("/cosma5/data/durham/dc-elbe1/hera/hera/geom_matching/wasserstein/wasserstein_dist " + topo_fname1 + " " + topo_fname2 + " >> " + outfile);
				time.sleep(0.5)

				#Clean up the files
				os.system("rm " + fname1 + " " + fname2)

			#Clean up the main files
			os.system("rm " + topo_fname1 + " " + topo_fname2)
