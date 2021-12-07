# referensi :https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/

from kalmanfilter import KalmanFilter
import dataset
import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter
kf = KalmanFilter()

# Dataset
read = np.genfromtxt("scene-0.txt")

data_vehicle = read[read[:,2] == 0].astype(float)
list_ins = list(dict.fromkeys(data_vehicle[:,1]))

for u in range(len(list_ins)):
    ins = data_vehicle[data_vehicle[:,1]==int(list_ins[u])]
    position = ins[:,[3,4]]

    #print(list_ins[u], len(position))

data_vehicle = read[read[:,1] == 11].astype(float)

print('jumlah frame: ',len(data_vehicle))

# Prediksi dan Visualisasi


for o in range(0, len(data_vehicle)):

    """plt.plot(0,"ko-", alpha=0.5, label='Historis')
    plt.plot(0,"yo", alpha=0.8, label='Current')
    plt.plot(0,"r*-", label='Prediction')
    plt.plot(0,"g^-", label='Ground Truth')
    plt.legend(loc="upper left")"""
    
    h,f,r,info = dataset.getData(o,10,6, data_vehicle)
    print('info ', info)
    print('Data historis ', h)
    print('Ground truth ',f)


    for x in h:
        predicted = kf.predict(x[0], x[1])
        #pred.append([predicted[0], predicted[1]])
        print('Prediksi - : ' , predicted)

        plt.plot(h[:,0], h[:,1], "ko-", alpha=0.5)
        plt.plot(h[:,0][-1], h[:,1][-1], "yo", alpha=0.8)
        plt.plot(predicted[0], predicted[1],"r*-")

    for _ in range(5):
        predicted = kf.predict(predicted[0], predicted[1])
        #pred.append([predicted[0], predicted[1]])
        print('Prediksi + : ', predicted)

        plt.plot(predicted[0], predicted[1],"r*-")

    
    plt.plot(f[:,0], f[:,1], "g^-")
    #plt.plot(np.array(pred)[:,0], np.array(pred)[:,0],"*-", label='Prediction')
    #plt.legend(loc="upper left")
    plt.xlabel('x-coord')
    plt.ylabel('y-coord')

    title = "id OV: {} - frame: {}".format( info[1], info[0])
    plt.title(title)
    
    name = "{}-{}.jpg".format(info[1], info[0])
    #plt.savefig(name)
    #plt.clf()
    #plt.show()
    
