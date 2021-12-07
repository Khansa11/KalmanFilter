# referensi: https://www.youtube.com/watch?v=m5Bw1m8jJuY

import numpy as np
import matplotlib.pyplot as plt

from kf import KF

##############
# Dataset
##############

read = np.genfromtxt("scene-0.txt")

data_vehicle = read[read[:,2] == 0].astype(float)
list_ins = list(dict.fromkeys(data_vehicle[:,1]))

for u in range(len(list_ins)):
    ins = data_vehicle[data_vehicle[:,1]==int(list_ins[u])]
    position = ins[:,[3,4]]

    #print(list_ins[u], len(position))

data_vehicle = read[read[:,1] == 11].astype(float)
data_vehicle = np.array(data_vehicle)[:,[3,4]]
print(data_vehicle)
dist = []
for i in range(len(data_vehicle)-1):
    
    ds = data_vehicle[i+1] - data_vehicle[i]
    jarak = np.sqrt((ds[0])**2 + (ds[1])**2)
    dist.append(jarak)
    
print(np.array(dist))
dist = np.array(dist)

##############
# Parameter
##############

meas_variance = 0.1 ** 2
DT = 0.5
mus = []
covs = []

##############
# Kalman Filter
##############

kf = KF(initial_x=0.0, initial_v=1.0, accel_variance=0.1) #hika acc = 0 maka grafik konstan
mus.append(kf.mean)
covs.append(kf.cov)

##############
# Predict and Update step
##############

for o in range(22):
    kf.predict(dt=DT)
    print('pred: ',kf.mean)
    kf.update(meas_value=dist[o], meas_variance=meas_variance)
    print('up:   ',kf.mean)

    mus.append(kf.mean)
    covs.append(kf.cov)

print(covs)

##############
# Visualisasi
##############

plt.title('Position')
plt.plot(dist, 'bo', label='Ground Truth')
plt.plot([mu[0] for mu in mus], 'r*', label='Prediction')
plt.legend(loc="upper left")
plt.xlabel('step number')
plt.ylabel('distance (m)')
plt.show()
