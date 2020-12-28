import numpy as np
import scipy.linalg
#from matplotlib import pyplot as plt

size = 16

model1a = "faint_opt"
model1b = "faint_opt"
model2a = "faint_opt"
model2b = "bright_opt"

Y1 = np.array([])
Y2 = np.array([])

for j in range(16):
	X1 = np.loadtxt("distances_" + model1a +"_" + model1b + "_" + str(j) + ".dat", usecols=range(1,8))
	X2 = np.loadtxt("distances_" + model2a +"_" + model2b + "_" + str(j) + ".dat", usecols=range(1,8))
	Y1 = np.append(X1,Y1)
	Y2 = np.append(X2,Y2)

Y1 = Y1.reshape((-1, 7))
Y2 = Y2.reshape((-1, 7))

seed1a = Y1[:,0]
seed1b = Y1[:,1]
slice1 = Y1[:,2]
noise1 = Y1[:,3]
sig1 = Y1[:,4]
dim1 =  Y1[:,5]
dist1 = Y1[:,6]

seed2a = Y2[:,0]
seed2b = Y2[:,1]
slice2 = Y2[:,2]
noise2 = Y2[:,3]
sig2 = Y2[:,4]
dim2 =  Y2[:,5]
dist2 = Y2[:,6]

noise_lvl = 0.0
signal_lvl = 1.0

model_uncertainty = 0.1

seeds = np.unique(np.append(Y1[:,0], Y1[:,1]))
ns = len(seeds)

L1 = np.zeros(ns)
L2 = np.zeros(ns)

for u in range(11):

	slices = np.array([u])
	# slices = np.arange(11)

	ndim = 3
	ndiags = ndim * len(slices)

	#Odd one out
	for i,seed in np.ndenumerate(seeds):

		#Observation matrix
		M = np.array([])
		#Retrieve the distances
		for d in range(ndim):
			for s in slices:
				D = Y1[(seed1a != seed)*(seed1b != seed)*(noise1 == noise_lvl)*(sig1 == signal_lvl)*(slice1 == s)*(dim1 == d),6]
				M = np.append(M,D)
		M = M.reshape((ndiags, -1))
		mean = M.mean(axis=1)

		#Estimate the covariance matrix
		H = M.dot(M.T) / len(M.T)
		H = H + model_uncertainty * np.diag(np.diag(H))
		# H = np.diag(np.diag(H))
		P = np.linalg.inv(H)
		fmat = np.sqrt(np.diag(1./np.diag(H)))
		corr = fmat.dot(H).dot(fmat)

		#Retrieve the distance with the singled out observation
		d_1 = np.array([])
		for d in range(ndim):
			for s in slices:
				D = Y1[((seed1a == seed) + (seed1b == seed))*(noise1 == noise_lvl)*(sig1 == signal_lvl)*(slice1 == s)*(dim1 == d),6]
				d_1 = np.append(d_1,D)
		d_1 = d_1.reshape((ndiags, -1))

		#From the second dataset, retrieve the distance with the singled out observation
		d_2 = np.array([])
		for d in range(ndim):
			for s in slices:
				D = Y2[(seed2b == seed)*(noise2 == noise_lvl)*(sig2 == signal_lvl)*(slice2 == s)*(dim2 == d),6]
				d_2 = np.append(d_2,D)
		d_2 = d_2.reshape((ndiags, -1))

		#Calculate the likelihoods
		L1[i] = np.exp(- 0.5 * np.diag(d_1.T.dot(P).dot(d_1))).mean()
		L2[i] = np.exp(- 0.5 * np.diag(d_2.T.dot(P).dot(d_2))).mean()
		# L1[i] = np.diag(scipy.linalg.expm(- 0.5 * d_1.T.dot(P).dot(d_1))).mean()
		# L2[i] = np.diag(scipy.linalg.expm(- 0.5 * d_2.T.dot(P).dot(d_2))).mean()

	p = L1/(L1+L2)
	m = p.mean()
	v = m*(1-m)

	llik = np.log(L1) - np.log(L1+L2)

	print(p.mean())

	# print(m, llik.mean(), (m - 0.5) / np.sqrt(v))

# print(p)
# print(m, m-np.sqrt(v), m+np.sqrt(v))

# a = np.exp(L1)
# b = np.exp(L2)
# p = a/(a+b)
# print(p)
# m=p.mean()
# v=m*(1-m)/16
# s=np.sqrt(v)
#
# print(m,m-s,m+s)
