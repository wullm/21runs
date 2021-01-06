import py21cmfast as p21c
from math import log10, sqrt
from mpi4py import MPI
import numpy as np
import h5py
from scipy.ndimage import zoom
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
def to_bytes_file(filename, arr, width):
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

#Prepare filename using a standard format
def generate_fname(dir, title, model, rank, seed, slice_num, pre_str, ext):
	fname = dir + "/" + title + "_" + model + "_" + str(rank) + "_" \
			+ str(seed) + "_slice_" + str(slice_num) + "_" + pre_str + ext
	return(fname)

#Compute the power spectrum
def compute_PS(grid, N, L, z):
	#Half the grid length rounded down
	Nhalf = int(N/2)

	#Calculate the wavevectors
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

	#Make a copy of the grid so as not to destroy the input
	grid_cp = grid * 1.0

	#Fourier transform the grid
	fgrid = np.fft.rfftn(grid_cp)
	fgrid = fgrid * (L*L*L) / (N*N*N)
	Pgrid = np.abs(fgrid)**2

	#Multiplicity of modes (double count all planes but z==0 and z==N/2)
	mult = np.ones_like(fgrid) * 2
	mult[:,:,0] = 1
	mult[:,:,-1] = 1

	#Compute the bin edges at the given redshift
	delta_k = dk_deta(z) * (1./bandwidth) * h # 1/Mpc
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
	#Convert to real numbers if Im(x) < eps
	C = np.real_if_close(C)

	return(C)

#Divide out the power spectrum from a grid
def whiten_grid(grid, N, L, z):
	#Compute the power spectrum
	C = compute_PS(grid, N, L, z)

	#Half the grid length rounded down
	Nhalf = int(N/2)

	#Calculate the wavevectors
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

	#Make a grid with the spherically averaged amplitude at each mode
	amplitude = np.sqrt(np.interp(k_cube.flatten(), C[:,0], C[:,2]))
	amplitude = amplitude.reshape((N,N,-1))

	#Make a copy of the grid so as not to destroy the input
	grid_cp = grid * 1.0

	#Fourier transform the grid
	fgrid = np.fft.rfftn(grid_cp)

	#Divide out the (square root of the) power spectrum
	fgrid_out = np.zeros_like(fgrid)
	nonzero = np.abs(amplitude) > 0
	fgrid_out[nonzero] = fgrid[nonzero] / amplitude[nonzero]

	return(fgrid_out)

#Compute the power spectrum and store it as a text file
def export_PS(grid, N, L, fname, z):
	#Compute the power spectrum
	C = compute_PS(grid, N, L, z)

	#Store the power spectrum data
	headr = """Power spectrum for brightness temperature at z=""" + str(z) + """
			\nWavenumbers k are in 1/Mpc, Delta^2(k) is in mK^2, P(k)
			is in mK^2 * Mpc^3\nThe power spectra are related by
			Delta^2(k) = P(k) * k^3 / (2*pi^2)\n\nk Delta^2_noise(k)
			P_noise(k)"""
	np.savetxt(fname, C, header=headr);

#We will load k-coverage cubes at different z's (observing time per day per k mode)
noise_data = {};

#The cut-off for the noise level at different z's
noise_cutoff = {};
cut = 0.8 # 80th percentile

#The redshifts for which we have telescope data
zvec = np.array([6,7,8,9,10,12,14,16,20,25,30]);
logz_vec = np.log(zvec);
nz = len(zvec);

for z in zvec:
	#Load the telescope data
	fname = "telescope_data/hera_350_k_coverage_z"+str(z)+".h5";
	f = h5py.File(fname, mode="r")
	noise_data[z] = f["Noise_Horizon"][:]
	#Compute the cut-off based on the noise cube without foreground avoidance
	full_noise = f["Noise"][:]
	noise_cutoff[z] = np.quantile(full_noise[full_noise > 0], cut)
	print("Noise cutoff at z = " + str(z) + " is " + str(noise_cutoff[z]))

#Grid dimensions
N = 128
L = 300 #Mpc
boxvol = L**3
Nhalf = int(N/2)

#Size of smaller downsized grids
N_small = 64

#Spacing of, minimum, and maximum wavenumbers for the full-sized grids
dk = 2*np.pi / L
k_min = dk
k_max = sqrt(3) * dk * N/2

#MPI dimensions
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#Model parameters for 21cmFAST
M_turn_faint = log10(0.5e9)
M_turn_bright = log10(7.5e9)
M_turn_moderate = log10(1.5e9)
f_star10_faint = log10(0.05)
f_star10_bright = log10(0.15)
f_star10_moderate = log10(0.07)

#The model to be run
model = "faint"

#Output directory
outdir = "faint"

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

#Each rank will work on a different random seed
seeds=np.array([605816, 150650, 278939, 573177, 691674, 773601, 727308, 828958,
				398521, 642050, 310828, 471108, 16985, 614889, 40760, 814979])
# seeds=np.array([359948, 830947, 694085, 785293, 862157, 213451, 382758, 775384,
#				 414375, 229164, 200363, 763514, 790364, 25155, 540756, 357669])
seed = seeds[rank];
print("Running seed", seed, rank, size);

#Shared parameters for 21cmFAST
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

#Compute the lightcone
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

print("Done with 21cm calculation")

#Retrieve the lightcone
arr = lightcone.brightness_temp # lightcone of size N * N * NZ
width = arr.shape[0] # N

#Dimension along the z dimension
Nz = arr.shape[2]
Lz = L/N*Nz #Mpc

#Calculate vectors of redshifts and comoving distances
z_vec = np.arange(6, 30, 0.01)
D_vec = lightcone.cosmo_params.cosmo.comoving_distance(z_vec).value

#Interpolate the redshifts along the z-dimension of the lightcone
zindex = np.arange(0, Nz, 1);
D = zindex / Nz * Lz
z = np.interp(D, D_vec - D_vec[0], z_vec)

#For N^3 cubes along the lightcone, calculate the wavevectors
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

#We will generate noise grids, linearly scaled by the noise level
noise_levels = np.array([0.0, 0.1, sqrt(0.1), 1.0, sqrt(10.0), 10.0, 1.0])
signal_levels = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

#Export all the cubic slices
slices = int(Nz/N)
for j in range(slices):
	#Retrieve a cubic grid of redshifts
	index_begin = j*N
	index_end = index_begin + N
	_,_,z_cube = np.meshgrid(np.ones(N), np.ones(N), z[index_begin:index_end])

	#The starting, ending, and central redshifts of this cube
	z_begin = z[index_begin];
	z_end = z[index_end];
	log_z_begin = np.log(z_begin)
	log_z_end = np.log(z_end)
	z_central = (z_begin+z_end)/2;

	#Find redshifts from the telescope data vector near z_begin and z_end
	z_near_begin = zvec[zvec <= z_begin].max()
	z_near_end = zvec[zvec >= z_end].min()
	log_z_near_begin = np.log(z_near_begin)
	log_z_near_end = np.log(z_near_end)

	#Get the theoretical signal
	signal = arr[:,:,index_begin:index_end];

	#Create a smaller copy
	small_signal = zoom(signal, zoom = 0.5, order = 1)

	# #Store the signal box without noise realization, horizon or resolution effects
	# signal_box_fname = generate_fname(outdir, "small", model, rank, seed, j, "noiseless", ".box")
	# to_bytes_file(signal_box_fname, small_signal, N_small)
	# signal_box_fname = generate_fname(outdir, "full", model, rank, seed, j, "noiseless", ".box")
	# to_bytes_file(signal_box_fname, signal, N)
	#
	# #Compute and store the power spectrum
	# PS_fname = generate_fname(outdir, "PS", model, rank, seed, j, "noiseless", ".dat")
	# export_PS(signal, N, L, PS_fname, z_central)

	#Fourier transform the signal
	fsignal = np.fft.rfftn(signal)

	#Generate a noise cube in Fourier space
	a = np.random.normal(0,1,(N, N, Nhalf+1))
	b = np.random.normal(0,1,(N, N, Nhalf+1))
	w = a + b*1j
	fgrf = w * np.sqrt(boxvol/2)

	#Retrieve the cut-off levels
	noise_cutoff_begin = noise_cutoff[z_near_begin]
	noise_cutoff_end = noise_cutoff[z_near_end]

	for noise_lvl, signal_lvl in zip(noise_levels, signal_levels):
		#Retrieve noise cubes from the telescope data at the bounding redshifts
		noise_cube_begin = noise_data[z_near_begin]
		noise_cube_end = noise_data[z_near_end]

		#Apply the cutoffs
		noise_cube_begin[noise_cube_begin * noise_lvl > noise_cutoff_begin] = 0
		noise_cube_end[noise_cube_end * noise_lvl > noise_cutoff_end] = 0

		#Apply the power spectra
		fgrf_begin = fgrf * np.sqrt(noise_cube_begin)
		fgrf_end = fgrf * np.sqrt(noise_cube_end)

		#Inverse Fourier transform (normalization needed)
		grf_begin = np.fft.irfftn(fgrf_begin) * N**3 / boxvol
		grf_end = np.fft.irfftn(fgrf_end) * N**3 / boxvol
		grf = np.zeros(N*N*N).reshape(N,N,N);

		#Zero out modes without uv coverage in the signal (these have zero noise)
		fsignal_begin = np.zeros_like(fsignal)
		fsignal_end = np.zeros_like(fsignal)
		fsignal_begin[noise_cube_begin > 0] = fsignal[noise_cube_begin > 0]
		fsignal_end[noise_cube_end > 0] = fsignal[noise_cube_end > 0]

		#Inverse Fourier transform (normalization not needed here)
		signal_begin = np.fft.irfftn(fsignal_begin)
		signal_end = np.fft.irfftn(fsignal_end)
		signal_recovered = np.zeros(N*N*N).reshape(N,N,N);

		#Interpolate between the two noise fields along the z-dimension
		for i in range(N):
			z_of_2d_slice = z[index_begin + i]
			logz = np.log(z_of_2d_slice)
			slice_begin = grf_begin[:,:,i]
			slice_end = grf_end[:,:,i]
			x = (logz - log_z_near_begin) / (log_z_near_end - log_z_near_begin)
			the_slice = slice_begin + (slice_end - slice_begin) * x
			grf[:,:,i] = the_slice

		#Interpolate between the two received signal fields along the z-dimension
		for i in range(N):
			z_of_2d_slice = z[index_begin + i]
			logz = np.log(z_of_2d_slice)
			slice_begin = signal_begin[:,:,i]
			slice_end = signal_end[:,:,i]
			x = (logz - log_z_near_begin) / (log_z_near_end - log_z_near_begin)
			the_slice = slice_begin + (slice_end - slice_begin) * x
			signal_recovered[:,:,i] = the_slice

		#Format the signal and noise levels into a string
		nsigstr = "noise_%.1f_signal_%.1f" % (noise_lvl, signal_lvl)

		#Add signal and noise with the appropriate levels
		total = signal_recovered * signal_lvl + grf * noise_lvl;
		#Fourier transform the total of signal and noise
		ftotal = np.fft.rfftn(total)

		#Also compute a whitened version of the grid
		ftotal_grey = whiten_grid(total, N, L, z_central)

		#And apply a Gaussian filter with smoothing radius R
		delta_nu = 240.0 / 1e6 # GHz
		R_smooth = delta_nu * 2.0 * np.pi / dk_deta(z_central) / h # Mpc
		ftotal = ftotal * np.exp(- 0.5 * R_smooth * R_smooth * k_cube * k_cube)
		ftotal_grey = ftotal_grey * np.exp(- 0.5 * R_smooth * R_smooth * k_cube * k_cube)

		print("Smooting radius R = ", R_smooth, "at z = ", z_central)

		#Inverse Fourier transform
		total = np.fft.irfftn(ftotal)
		total_grey = np.fft.irfftn(ftotal_grey)

		#Discard invalid points to be sure (rare)
		total[np.isnan(total)] = 0.
		total_grey[np.isnan(total_grey)] = 0.

		#Create smaller copies
		small_total = zoom(total, zoom = 0.5, order = 1)
		small_total_grey = zoom(total_grey, zoom = 0.5, order = 1)

		#Demean each redshift slice
		total -= total.mean(axis=2, keepdims=True)
		total_grey -= total_grey.mean(axis=2, keepdims=True)
		small_total -= small_total.mean(axis=2, keepdims=True)
		small_total_grey -= small_total_grey.mean(axis=2, keepdims=True)

		#Store the box with noise
		# box_fname = generate_fname(outdir, "small", model, rank, seed, j, nsigstr, ".box")
		# to_bytes_file(box_fname, small_total, N_small)
		# box_fname = generate_fname(outdir, "full", model, rank, seed, j, nsigstr, ".box")
		# to_bytes_file(box_fname, total, N)

		#Store image of a 2D slice of the 3D cube
		# image_fname = generate_fname(outdir, "xy_thick", model, rank, seed, j, nsigstr, ".png")
		# plt.imsave(image_fname, total[:,:,22:42].mean(axis=2), cmap="magma")

		#Store the box with noise (whitened)
		box_fname = generate_fname(outdir, "small_grey", model, rank, seed, j, nsigstr, ".box")
		to_bytes_file(box_fname, small_total_grey, N_small)
		box_fname = generate_fname(outdir, "full_grey", model, rank, seed, j, nsigstr, ".box")
		to_bytes_file(box_fname, total_grey, N)

		print("Stored grid for slice ", j, " and noise level ", noise_lvl);

		#Store the power spectrum data
		PS_fname = generate_fname(outdir, "PS", model, rank, seed, j, nsigstr, ".dat")
		export_PS(total, N, L, PS_fname, z_central)

		#Store the power spectrum data for the white grid
		PS_grey_fname = generate_fname(outdir, "PS_grey", model, rank, seed, j, nsigstr, ".dat")
		export_PS(total_grey, N, L, PS_grey_fname, z_central)
		print("Done with power spectrum for slice ", j, " and noise level ", noise_lvl);
