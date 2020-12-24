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

#Hubble constant
h = 0.68

#####################
#Two methods from 21cmsense by J Pober used for bandwith -> wavenumber calculation
#####################

#Multiply by this to convert a bandwidth in GHz to a line of sight distance in Mpc/h at redshift z
def dL_df(z, omega_m=0.31):
    '''[h^-1 Mpc]/GHz, from Furlanetto et al. (2006)'''
    return (1.7 / 0.1) * ((1+z) / 10.)**.5 * (omega_m/0.15)**-0.5 * 1e3

#Multiply by this to convert eta (FT of freq.; in 1/GHz) to line of sight k mode in h/Mpc at redshift z
def dk_deta(z):
    '''2pi * [h Mpc^-1] / [GHz^-1]'''
    return 2*np.pi / dL_df(z)

#####################
#End of methods from 21cmsense
#####################

#The fiducial bandwidth in GHz
bandwidth = 0.008

#Function to save data cubes to the disk
def to_bytes_file(filename, arr):
	with open(filename, "wb") as f:
		size = arr.size;
		for i in range(size):
			z = i % width;
			y = int(i/width) % width;
			x = int(i/(width*width)) % width;
			the_float = arr[x,y,z];
			four_bytes = struct.pack('f', the_float);
			f.write(four_bytes);
		f.close();


#We will load k-coverage cubes at different z's (observing time per day per k mode)
noise_data = {};

#The redshifts
zvec = np.array([6,7,8,9,10,12,14,16,20,25,30]);
logz_vec = np.log(zvec);

#Load the data
for i,z in enumerate(zvec):
	fname = "telescope_data/hera_350_k_coverage_z"+str(z)+".h5";
	f = h5py.File(fname, mode="r")
	noise_data[z] = f["Noise_Horizon"][:]

#Number of redshifts
nz = len(zvec);

#Grid dimensions
N = 128
L = 300 #Mpc
boxvol = L**3
Nhalf = int(N/2)

dk = 2*np.pi / L
k_min = dk
k_max = sqrt(3) * dk * N/2

#Now, we will compute 21cm lightcones with 21cmFAST
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#Model parameters
M_turn_faint = log10(0.5e9)
M_turn_bright = log10(7.5e9)
M_turn_moderate = log10(1.5e9)
f_star10_faint = log10(0.05)
f_star10_bright = log10(0.15)
f_star10_moderate = log10(0.07)

#The model to be run
model = "faint"

#Apply model parameters
if model == "faint":
	M_turn = M_turn_faint
	f_star10 = f_star10_faint
elif model == "moderate":
	M_turn = M_turn_moderate
	f_star10 = f_star10_moderate
else:
	M_turn = M_turn_bright
	f_star10 = f_star10_bright

runs = 1

seeds=np.array([605816,150650,278939,573177,691674,773601,727308,828958,398521,642050,310828,471108,16985,614889,40760,814979])

for run in range(runs):
	#seed=int(round(1e6*random()));
	seed = seeds[rank];
	print("Random seed", seed, rank, size);

	astro_pars = p21c.AstroParams({
		"ALPHA_STAR": 0.5,
		"ALPHA_ESC": -0.5,
		"F_ESC10": log10(0.1),
		"t_STAR": 0.5,
		"L_X": 40.5,
		"NU_X_THRESH": 500,
		"M_TURN": M_turn,
		"F_STAR10": f_star10
	})

	cosmo_pars = p21c.CosmoParams({
		"SIGMA_8": 0.81,
		"hlittle": h,
		"OMm": 0.31,
		"OMb": 0.048,
		"POWER_INDEX": 0.97
	})

	flag_pars = p21c.FlagOptions({
		"INHOMO_RECO": True,
		"USE_MASS_DEPENDENT_ZETA": True,
		"PHOTON_CONS": False,
		"USE_MINI_HALOS": False,
		"USE_TS_FLUCT": True
	})

	user_pars = {
		"HII_DIM": N,
		"DIM": 4*N,
		"BOX_LEN": L, #Mpc
		"USE_INTERPOLATION_TABLES": True
	}

	lightcone = p21c.run_lightcone(
		redshift = 6.0,
		max_redshift = 25.0,
		user_params = user_pars,
		cosmo_params = cosmo_pars,
		astro_params = astro_pars,
		flag_options = flag_pars,
		random_seed = seed,
		lightcone_quantities=("brightness_temp", "density", "xH_box"),
		global_quantities=("brightness_temp", "density", "xH_box")
	)

	print("Done with 21cm")

	arr = lightcone.brightness_temp # lightcone of size N * N * NZ
	width = arr.shape[0] # N

	#Dimension along the z dimension
	Nz = arr.shape[2]
	Lz = L/N*Nz #Mpc

	#Calculate vectors of redshifts and comoving distances
	z_vec = np.arange(6,30,0.01)
	D_vec = lightcone.cosmo_params.cosmo.comoving_distance(z_vec).value

	#Interpolate the redshifts along the z-dimension of the lightcone
	zindex = np.arange(0,Nz,1);
	D = zindex / Nz * Lz
	z = np.interp(D, D_vec - D_vec[0], z_vec)

	#For N^3 cubes along the lightcone, calculate the wavevectors vectors
	dk = 2*np.pi / L
	modes = np.arange(N)
	modes[modes > N/2] -= N
	kx = modes * dk
	ky = modes * dk
	kz = modes[:Nhalf+1] * dk

	#Calculate the N*N*(N/2+1) grid of wavenumbers
	KX,KY,KZ = np.meshgrid(kx,ky,kz)
	K2 = KX**2 + KY**2 + KZ**2
	k_cube = np.sqrt(K2)

	#We will generate noise grids, linearly scaled by the noise level between (0,2)
	noise_levels = np.arange(0,2.1,0.1);
	#noise_levels = np.array([3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]);

	#Export all the cubic slices
	slices = int(Nz/N)
	for j in range(slices):
		index_begin = j*N
		index_end = index_begin + N
		_,_,z_cube = np.meshgrid(np.ones(N), np.ones(N), z[index_begin:index_end])

		z_begin = z[index_begin];
		z_end = z[index_end];
		log_z_begin = np.log(z_begin)
		log_z_end = np.log(z_end)
		z_central = (z_begin+z_end)/2;

		#Find redshifts from the limited data vector near z_begin and z_end
		z_near_begin = zvec[zvec <= z_begin].max()
		z_near_end = zvec[zvec >= z_end].min()
		log_z_near_begin = np.log(z_near_begin)
		log_z_near_end = np.log(z_near_end)

		#Get the theoretical signal
		signal = arr[:,:,index_begin:index_end];

		#Store the signal box without noise realization
		signal_box_fname = model + "/dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noiseless.box";
		to_bytes_file(signal_box_fname, signal)

		#Store images of a 2D slice of the 3D cube
		image_fname = model + "/dT_xy_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noiseless.png";
		plt.imsave(image_fname, signal[:,:,32], cmap="magma")
		image_fname = model + "/dT_xy_thick_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noiseless.png";
		plt.imsave(image_fname, signal[:,:,22:42].mean(axis=2), cmap="magma")
		image_fname = model + "/dT_yz_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noiseless.png";
		plt.imsave(image_fname, signal[32], cmap="magma")

		#Fourier transform the signal
		fsignal = np.fft.rfftn(signal)

		# Generate a noise cube in Fourier space
		a = np.random.normal(0,1,(N, N, Nhalf+1))
		b = np.random.normal(0,1,(N, N, Nhalf+1))
		w = a + b*1j
		fgrf = w * np.sqrt(boxvol/2)

		#Apply the noise cube
		noise_cube_begin = noise_data[z_near_begin]
		noise_cube_end = noise_data[z_near_end]

		#Apply the power spectra
		fgrf_begin = fgrf * np.sqrt(noise_cube_begin)
		fgrf_end = fgrf * np.sqrt(noise_cube_end)

		#Inverse Fourier transform
		grf_begin = np.fft.irfftn(fgrf_begin) * N**3 / boxvol
		grf_end = np.fft.irfftn(fgrf_end) * N**3 / boxvol
		grf = np.zeros(N*N*N).reshape(N,N,N);

		#Zero out modes without uv coverage in the signal
		fsignal_begin = np.zeros_like(fsignal)
		fsignal_end = np.zeros_like(fsignal)
		fsignal_begin[noise_cube_begin > 0] = fsignal[noise_cube_begin > 0]
		fsignal_end[noise_cube_end > 0] = fsignal[noise_cube_end > 0]

		#Inverse Fourier transform
		signal_begin = np.fft.irfftn(fsignal_begin)
		signal_end = np.fft.irfftn(fsignal_end)
		signal_recovered = np.zeros(N*N*N).reshape(N,N,N);

		#Interpolate between the two noise fields along the z-dimension
		for i in range(N):
			thez = z[index_begin + i]
			logz = np.log(thez)
			slice_begin = grf_begin[:,:,i]
			slice_end = grf_end[:,:,i]
			the_slice = slice_begin + (slice_end - slice_begin)/(log_z_near_end - log_z_near_begin) * (logz - log_z_near_begin)
			grf[:,:,i] = the_slice

		#Interpolate between the two received signal fields along the z-dimension
		for i in range(N):
			thez = z[index_begin + i]
			logz = np.log(thez)
			slice_begin = signal_begin[:,:,i]
			slice_end = signal_end[:,:,i]
			the_slice = slice_begin + (slice_end - slice_begin)/(log_z_near_end - log_z_near_begin) * (logz - log_z_near_begin)
			signal_recovered[:,:,i] = the_slice

		#Compute the pure noise power spectrum (for noise_lvl 1.0)

		#Store the pure noise box
		box_fname = model + "/dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_pure_noise.box";
		to_bytes_file(box_fname, grf)

		#Store images of a 2D slice of the pure noise cube
		image_fname = model + "/dT_xy_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_pure_noise.png";
		plt.imsave(image_fname, grf[:,:,32], cmap="magma")
		image_fname = model + "/dT_xy_thick_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_pure_noise.png";
		plt.imsave(image_fname, grf[:,:,22:42].mean(axis=2), cmap="magma")
		image_fname = model + "/dT_yz_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_pure_noise.png";
		plt.imsave(image_fname, grf[32], cmap="magma")

		#Prepare grid for power spectrum calculation
		grid = grf * 1.0

		#Fourier transform the grid
		fgrid = np.fft.rfftn(grid)
		fgrid = fgrid * (L*L*L) / (N*N*N)
		Pgrid = np.abs(fgrid)**2

		#Multiplicity of modes (double count the planes with z==0 and z==N/2)
		mult = np.ones_like(fgrid) * 2
		mult[:,:,0] = 1
		mult[:,:,-1] = 1

		#Compute the bin edges at the central redshift
		delta_k = dk_deta(z_central) * (1./bandwidth) * h # 1/Mpc
		kvec = np.arange(delta_k, k_max, delta_k) # 1/Mpc
		bin_edges = np.zeros(len(kvec)+1)
		bin_edges[0] = k_min
		bin_edges[-1] = k_max
		bin_edges[1:-1] = 0.5 * (kvec[1:] + kvec[:-1])

		#Compute the power spectrum
		obs = np.histogram(k_cube, bin_edges, weights = mult)[0]
		Pow = np.histogram(k_cube, bin_edges, weights = Pgrid * mult)[0]
		avg_k = np.histogram(k_cube, bin_edges, weights = k_cube * mult)[0]

		#Normalization
		Pow = Pow / obs
		Pow = Pow / (L*L*L)
		avg_k = avg_k / obs

		#Convert to real numbers if Im(x) < eps
		Pow = np.real_if_close(Pow)
		avg_k = np.real_if_close(avg_k)
		obs = np.real_if_close(obs)

		#Convert to "dimensionless" (has dimensions mK^2) power spectrum
		Delta2 = Pow * avg_k**3 / (2*np.pi**2)
		B = np.array([avg_k, Delta2, Pow]).T
		C = B[np.isnan(avg_k) == False,:]

		#Store the power spectrum data
		PS_fname = model + "/PS_dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_pure_noise.dat";
		np.savetxt(PS_fname, C, header="Power spectrum for brightness temperature (pure noise) at z="+str(z_central)+"\nWavenumbers k are in 1/Mpc, Delta^2(k) is in mK^2, P(k) is in mK^2 * Mpc^3\nThe power spectra are related by Delta^2(k) = P(k) * k^3 / (2*pi^2)\n\nk Delta^2_noise(k) P_noise(k)");

		for noise_lvl in noise_levels:
			#Add the noise with the noise level
			total = signal_recovered + grf * noise_lvl;

			#Finally apply a sharp k-space filter on k in (0.1, 1.0)
			ftotal = np.fft.rfftn(total)
			ftotal[k_cube > 1.0] = 0.0
			# ftotal[k_cube < 0.1] = 0.0
			#And apply a Gaussian filter with smoothing radius of 1 Mpc
			#ftotal = ftotal * np.exp(-k_cube * k_cube)
			total = np.fft.irfftn(ftotal)

			#Discard invalid points (rare)
			total[np.isnan(total)] = 0.

			#Store the box with noise
			box_fname = model + "/dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".box";
			to_bytes_file(box_fname, total)

			#Store images of a 2D slice of the 3D cube
			image_fname = model + "/dT_xy_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".png";
			plt.imsave(image_fname, total[:,:,32], cmap="magma")
			image_fname = model + "/dT_xy_thick_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".png";
			plt.imsave(image_fname, total[:,:,22:42].mean(axis=2), cmap="magma")
			image_fname = model + "/dT_yz_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".png";
			plt.imsave(image_fname, total[32], cmap="magma")


			print("Stored grid for slice ", j, " and noise level ", noise_lvl);

			#Prepare grid for power spectrum calculation
			grid = total * 1.0

			#Fourier transform the grid
			fgrid = np.fft.rfftn(grid)
			fgrid = fgrid * (L*L*L) / (N*N*N)
			Pgrid = np.abs(fgrid)**2

			#Multiplicity of modes (double count the planes with z==0 and z==N/2)
			mult = np.ones_like(fgrid) * 2
			mult[:,:,0] = 1
			mult[:,:,-1] = 1

			#Compute the power spectrum
			obs = np.histogram(k_cube, bin_edges, weights = mult)[0]
			Pow = np.histogram(k_cube, bin_edges, weights = Pgrid * mult)[0]
			avg_k = np.histogram(k_cube, bin_edges, weights = k_cube * mult)[0]

			#Normalization
			Pow = Pow / obs
			Pow = Pow / (L*L*L)
			avg_k = avg_k / obs

			#Convert to real numbers if Im(x) < eps
			Pow = np.real_if_close(Pow)
			avg_k = np.real_if_close(avg_k)
			obs = np.real_if_close(obs)

			#Convert to "dimensionless" (has dimensions mK^2) power spectrum
			Delta2 = Pow * avg_k**3 / (2*np.pi**2)
			B = np.array([avg_k, Delta2, Pow]).T
			C = B[np.isnan(avg_k) == False,:]

			#Store the power spectrum data
			PS_fname = model + "/PS_dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".dat";
			np.savetxt(PS_fname, C, header="Power spectrum for brightness temperature at z="+str(z_central)+"\nWavenumbers k are in 1/Mpc, Delta^2(k) is in mK^2, P(k) is in mK^2 * Mpc^3\nThe power spectra are related by Delta^2(k) = P(k) * k^3 / (2*pi^2)\n\nk Delta^2_noise(k) P_noise(k)");

			print("Done with power spectrum for slice ", j, " and noise level ", noise_lvl);

			#Finally compute the topology
			#topology_fname = model + "/topology_dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".dat";
			#os.system("/cosma5/data/durham/dc-elbe1/FieldFiltrations/FieldFiltrations/triangulate " + box_fname + " > " + topology_fname);
			#print("Done with topology for slice ", j, " and noise level ", noise_lvl);
