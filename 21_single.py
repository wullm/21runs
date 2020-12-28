import py21cmfast as p21c
from math import log10, sqrt
import numpy as np
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

#Grid dimensions
N = 128
L = 300 #Mpc
boxvol = L**3
Nhalf = int(N/2)

#Spacing of, minimum, and maximum wavenumbers for the full-sized grids
dk = 2*np.pi / L
k_min = dk
k_max = sqrt(3) * dk * N/2

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

#No MPI
rank = 0
size = 1

#Each rank will work on a different random seed
seeds=np.array([605816, 150650, 278939, 573177, 691674, 773601, 727308, 828958,
				398521, 642050, 310828, 471108, 16985, 614889, 40760, 814979])
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

    print(j, z_begin, z_central, z_end)
