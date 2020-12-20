import py21cmfast as p21c
from math import log10, sqrt
from matplotlib import pyplot as plt
from mpi4py import MPI
from random import random

import ctypes

import numpy as np
from scipy import interpolate as interp

import struct; #for bytes to float
import scipy.ndimage;
from scipy import misc;
import os;

#Library with a power spectrum tool
lib = ctypes.cdll.LoadLibrary('/cosma5/data/durham/dc-elbe1/montep/meshpt/meshpt.so')

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


#We will load noise data arrays P(k) at different redshifts z
noise_data = {};

#The redshifts
zvec = np.array([6,7,8,9,10,12,14,16,20,25]);
logz_vec = np.log(zvec);

#The wavenumbers are different per file, so we will interpolate on a finer k-grid
logk_vec = np.log(10) * np.arange(-1.1,0.4,0.001);
kvec = np.exp(logk_vec);

#Dimensions of the final noise array, which will be used in later interpolations
nz = len(zvec);
nk = len(kvec);

#The power spectrum is approximately a power law in both k and z
#We will interpolate log(P_noise) linearly on (log_z, log_k)
#Create a 2D array log_P_noise(z,k)
logP_arr = np.zeros(nz * nk).reshape(nz, nk);

#Sample a random realization of the noise power spectrum
df = 2 # two degrees of freedom ensures that for X=chi^2/df, mu_X = sigma_X^2 = 2.0
chi2 = np.random.chisquare(df, size = 100) / df;

#Load the data and create the finer 2D grid
for i,z in enumerate(zvec):
	fname = "telescope_noise/hera_331_noise_z"+str(z)+".txt";
	noise_data[z] = np.loadtxt(fname)
	k = noise_data[z][:,0];
	P = noise_data[z][:,2];
	#Sample a random realization of the noise power spectrum
	#df = 2 # two degrees of freedom ensures that for X=chi^2/df, mu_X = sigma_X^2 = 2.0
	#chi2 = np.random.chisquare(df, size = len(P)) / df;
	P = P * chi2[:len(P)];
	f_interp = interp.interp1d(np.log(k), np.log(P), kind="nearest", bounds_error=False, fill_value="extrapolate")
	logP_arr[i,:] = f_interp(logk_vec)
	#logP_arr[i,:] = np.interp(logk_vec, np.log(k), np.log(P))

#Function log P(log(k), log(z))
logP_func = interp.interp2d(logk_vec, logz_vec, logP_arr, kind="linear")
#logkM, logzM = np.meshgrid(logk_vec, logz_vec)
#logP_func = interp.NearestNDInterpolator(logkM.flatten(), logzM.flatten(), logP_arr)
#Function P(k,z)
P_func = lambda k,z: np.exp(logP_func(np.log(k),np.log(z)))

print("so far so good")


#We will load telescope data arrays at different redshifts z
telescope_data = {};

#The power spectrum is approximately a power law in both k and z
#We will interpolate log(P_noise) linearly on (log_z, log_k) 
#Create a 2D array log_P_noise(z,k)
SIZE = 39 # telescope uv grid size
horizon_arr = np.zeros(nz * SIZE * SIZE).reshape(nz, SIZE, SIZE);
uv_arr = np.zeros(nz * SIZE * SIZE).reshape(nz, SIZE, SIZE);
dk_vec = np.zeros(nz)
T2_vec = np.zeros(nz)

#Load the data
for i,z in enumerate(zvec):
        fname1 = "telescope_data/hera_331_horizon_z"+str(z)+".txt";
        fname2 = "telescope_data/hera_331_uv_coverage_z"+str(z)+".txt";
        fname3 = "telescope_data/hera_331_numbers_z"+str(z)+".txt";
        telescope_data[z] = {}
        telescope_data[z]["horizon"] = np.loadtxt(fname1)
        telescope_data[z]["uv_coverage"] = np.loadtxt(fname2)
        telescope_data[z]["numbers"] = np.loadtxt(fname3)
        horizon_arr[i,:,:] = telescope_data[z]["horizon"]
        uv_arr[i,:,:] = telescope_data[z]["uv_coverage"]
        dk_vec[i] = telescope_data[z]["numbers"][1]
        T2_vec[i] = telescope_data[z]["numbers"][2]

#Grid dimensions
N = 128
L = 300 #Mpc
boxvol = L**3
Nhalf = round(N/2+1)

#Convert to c types for later
cN = ctypes.c_int(N)
cL = ctypes.c_double(L)

#Now, we will compute 21cm lightcones with 21cmFAST
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

M_turn_faint = log10(0.5e9)
M_turn_bright = log10(7.5e9)
M_turn_moderate = log10(1.5e9)
f_star10_faint = log10(0.05)
f_star10_bright = log10(0.15)
f_star10_moderate = log10(0.07)

model = "moderate"

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
		"hlittle": 0.68,
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
	kx = np.roll(np.arange(-Nhalf+2,Nhalf), Nhalf) * dk
	ky = np.roll(np.arange(-Nhalf+2,Nhalf), Nhalf) * dk
	kz = np.arange(0,Nhalf) * dk

	#Calculate the N*N*Nhalf grid of wavenumbers
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

		#Get the theoretical signal
		signal = arr[:,:,index_begin:index_end];

		#Store the signal box without noise realization
		#signal_box_fname = model + "/dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noiseless.box";
		#to_bytes_file(signal_box_fname, signal)

		# Generate a noise cube in Fourier space
		a = np.random.normal(0,1,N*N*Nhalf).reshape(N,N,Nhalf)
		b = np.random.normal(0,1,N*N*Nhalf).reshape(N,N,Nhalf)
		w = a + b*1j
		fgrf = w * np.sqrt(boxvol/2)

		#Apply the power spectrum, appropriately interpolated on (k,z)
		P_cube_begin = np.zeros(N*N*Nhalf).reshape(N,N,Nhalf)
		P_cube_end = np.zeros(N*N*Nhalf).reshape(N,N,Nhalf)

		#Interpolate the power spectrum to the grid for z_begin
		for i in range(Nhalf):
			thek = k_cube[:,:,i]
			interp_P = logP_func(logk_vec, log_z_begin)
			theLogP = np.interp(np.log(thek).flatten(), logk_vec, interp_P)
			theP = np.exp(theLogP).reshape(N,N)
			P_cube_begin[:,:,i] = theP

		#Interpolate the power spectrum to the grid for z_end
		for i in range(Nhalf):
			thek = k_cube[:,:,i]
			interp_P = logP_func(logk_vec, log_z_end)
			theLogP = np.interp(np.log(thek).flatten(), logk_vec, interp_P)
			theP = np.exp(theLogP).reshape(N,N)
			P_cube_end[:,:,i] = theP

		observable_begin = np.zeros(N*N*Nhalf).reshape(N,N,Nhalf)
		observable_end = np.zeros(N*N*Nhalf).reshape(N,N,Nhalf)

		#Interpolate the power spectrum to the grid for z_begin
		for i in range(Nhalf):
			#Wavevectors in the orthogonal direction to the line of sight
			plane_kx = KX[:,:,i]
			plane_ky = KY[:,:,i]
			#Quantities corresponding to the current redshift
			plane_dk = 1.0/0.15/2.0*np.interp(log_z_begin, logz_vec, dk_vec)
			plane_T2 = np.interp(log_z_begin, logz_vec, T2_vec)
			plane_kz = kz[i]
			#Interpolate the telescope data to the current redshift plane
			index = np.interp(log_z_begin, logz_vec, np.arange(nz))
			if (index >= nz):
				uv_plane = uv_arr[-1]
				horizon_plane = horizon_arr[-1]
			else:
				uv_plane = uv_arr[int(index)] + (uv_arr[int(index)+1] - uv_arr[int(index)]) * (index - int(index))
				horizon_plane = horizon_arr[int(index)] + (horizon_arr[int(index)+1] - horizon_arr[int(index)]) * (index - int(index))
			#Rescale the planes to the appropriate sizes
			kxvec = plane_dk * (np.arange(SIZE) - int(SIZE/2));
			kyvec = plane_dk * (np.arange(SIZE) - int(SIZE/2));
			uv_func = interp.interp2d(kxvec, kyvec, uv_plane)
			horizon_func = interp.interp2d(kxvec, kyvec, horizon_plane)
			#Insert into the larger plane
			large_uv_plane = uv_func(kx, ky)
			large_horizon_plane = horizon_func(kx, ky)
			#Make sure that uv coverage is conserved
			large_uv_plane *= uv_plane.sum() / large_uv_plane.sum()
			#Roll the arrays over to match our conventions
			large_uv_plane = np.roll(large_uv_plane, (Nhalf, Nhalf), axis=(0,1))
			large_horizon_plane = np.roll(large_horizon_plane, (Nhalf, Nhalf), axis=(0,1))
			#Compute the noise temperature grid
			large_T2_grid = np.zeros(N*N).reshape(N,N)
			large_T2_grid[large_uv_plane > 0] = plane_T2 / large_uv_plane[large_uv_plane > 0]
			#Construct binary grid according to whether the mode is observable
			large_observable_grid = np.zeros(N*N).reshape(N,N)
			large_observable_grid[large_T2_grid > 0] = 1
			observable_begin[:,:,i] = large_observable_grid

		#Interpolate the power spectrum to the grid for z_end
		for i in range(Nhalf):
                        #Wavevectors in the orthogonal direction to the line of sight
                        plane_kx = KX[:,:,i]
                        plane_ky = KY[:,:,i]
                        #Quantities corresponding to the current redshift
                        plane_dk = 1.0/0.15/2.0*np.interp(log_z_end, logz_vec, dk_vec)
                        plane_T2 = np.interp(log_z_begin, logz_vec, T2_vec)
                        plane_kz = kz[i]
                        #Interpolate the telescope data to the current redshift plane
                        index = np.interp(log_z_begin, logz_vec, np.arange(nz))
                        if (index >= nz):
                                uv_plane = uv_arr[-1]
                                horizon_plane = horizon_arr[-1]
                        else:
                                uv_plane = uv_arr[int(index)] + (uv_arr[int(index)+1] - uv_arr[int(index)]) * (index - int(index))
                                horizon_plane = horizon_arr[int(index)] + (horizon_arr[int(index)+1] - horizon_arr[int(index)]) * (index - int(index))
                        #Rescale the planes to the appropriate sizes
                        kxvec = plane_dk * (np.arange(SIZE) - int(SIZE/2));
                        kyvec = plane_dk * (np.arange(SIZE) - int(SIZE/2));
                        uv_func = interp.interp2d(kxvec, kyvec, uv_plane)
                        horizon_func = interp.interp2d(kxvec, kyvec, horizon_plane)
                        #Insert into the larger plane
                        large_uv_plane = uv_func(kx, ky)
                        large_horizon_plane = horizon_func(kx, ky)
                        #Make sure that uv coverage is conserved
                        large_uv_plane *= uv_plane.sum() / large_uv_plane.sum()
                        #Roll the arrays over to match our conventions
                        large_uv_plane = np.roll(large_uv_plane, (Nhalf, Nhalf), axis=(0,1))
                        large_horizon_plane = np.roll(large_horizon_plane, (Nhalf, Nhalf), axis=(0,1))
                        #Compute the noise temperature grid
                        large_T2_grid = np.zeros(N*N).reshape(N,N)
                        large_T2_grid[large_uv_plane > 0] = plane_T2 / large_uv_plane[large_uv_plane > 0]
                        #Construct binary grid according to whether the mode is observable
                        large_observable_grid = np.zeros(N*N).reshape(N,N)
                        large_observable_grid[large_T2_grid > 0] = 1
                        observable_end[:,:,i] = large_observable_grid

		#Apply the observable windows to the noise
		P_cube_begin *= observable_begin;
		P_cube_end *= observable_end;

		#Apply the power spectra
		fgrf_begin = fgrf * np.sqrt(P_cube_begin)
		fgrf_end = fgrf * np.sqrt(P_cube_end)

		#Apply the observable windows to the signal
		fsig = np.fft.rfftn(signal)
		fsig_begin = fsig * observable_begin
		fsig_end = fsig * observable_end
		sig_begin = np.fft.irfftn(fsig_begin)
		sig_end = np.fft.irfftn(fsig_end)
		windowed_signal = np.zeros(N*N*N).reshape(N,N,N)

		#Inverse Fourier transform
		grf_begin = np.fft.irfftn(fgrf_begin) * N**3 / boxvol
		grf_end = np.fft.irfftn(fgrf_end) * N**3 / boxvol
		grf = np.zeros(N*N*N).reshape(N,N,N);

		#Interpolate between the two noise fields along the z-dimension
		for i in range(N):
			thez = z[index_begin + i]
			logz = np.log(thez)
			slice_begin = grf_begin[:,:,i]
			slice_end = grf_end[:,:,i]
			the_slice = slice_begin + (slice_end - slice_begin)/(log_z_end - log_z_begin) * (logz - log_z_begin)
			grf[:,:,i] = the_slice
			
			sig_slice_begin = sig_begin[:,:,i]
			sig_slice_end = sig_end[:,:,i]
			sig_slice = sig_slice_begin + (sig_slice_end - sig_slice_begin)/(log_z_end - log_z_begin) * (logz - log_z_begin)
			windowed_signal[:,:,i] = sig_slice

		for noise_lvl in noise_levels:
			#Add the noise with the noise level
			total = windowed_signal + grf * noise_lvl;

			#Finally apply a sharp k-space filter on k in (0.1, 1.0)
			ftotal = np.fft.rfftn(total)
			ftotal[k_cube > 1.0] = 0.0
			ftotal[k_cube < 0.1] = 0.0
			#And apply a Gaussian filter with smoothing radius of 1 Mpc
			#ftotal = ftotal * np.exp(-k_cube * k_cube)
			total = np.fft.irfftn(ftotal)

			#Discard invalid points (rare)
			total[np.isnan(total)] = 0.

			#Store the box with noise
			box_fname = model + "/dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".box";
			to_bytes_file(box_fname, total)

			print("Stored grid for slice ", j, " and noise level ", noise_lvl);

			#Prepare grid for power spectrum calculation
			grid = total * 1.0
			c_grid = ctypes.c_void_p(grid.ctypes.data);

			#Create bins and empty vectors for power spectrum calculation
			bins = 30
			k_in_bins = np.zeros(bins);
			P_in_bins = np.zeros(bins);
			obs_in_bins = np.zeros(bins).astype(int);

			c_bins = ctypes.c_int(bins);
			c_kbins = ctypes.c_void_p(k_in_bins.ctypes.data);
			c_Pbins = ctypes.c_void_p(P_in_bins.ctypes.data);
			c_obins = ctypes.c_void_p(obs_in_bins.ctypes.data);

			#Compute the power spectrum
			lib.gridPowerSpec(cN, cL, c_bins, c_grid, c_kbins, c_Pbins, c_obins);

			#Convert to "dimensionless" (has dimensions mK^2) power spectrum
			Delta2 = P_in_bins * k_in_bins**3 / (2*np.pi**2)
			B = np.array([k_in_bins, Delta2, P_in_bins]).T
			C = B[np.isnan(k_in_bins) == False,:]

			#Store the power spectrum data
			PS_fname = model + "/PS_dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".dat";
			np.savetxt(PS_fname, C, header="Power spectrum for brightness temperature at z="+str(z_central)+"\nWavenumbers k are in 1/Mpc, Delta^2(k) is in mK^2, P(k) is in mK^2 * Mpc^3\nThe power spectra are related by Delta^2(k) = P(k) * k^3 / (2*pi^2)\n\nk Delta^2_noise(k) P_noise(k)");

			print("Done with power spectrum for slice ", j, " and noise level ", noise_lvl);

			#Finally compute the topology
			#topology_fname = model + "/topology_dT_" + model + "_" + str(rank) + "_" + str(seed) + "_slice_" + str(j) + "_noise_" + str(round(noise_lvl,1)) + ".dat";
			#os.system("/cosma5/data/durham/dc-elbe1/FieldFiltrations/FieldFiltrations/triangulate " + box_fname + " > " + topology_fname);
			#print("Done with topology for slice ", j, " and noise level ", noise_lvl);
