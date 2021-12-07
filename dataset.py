# referensi: https://github.com/BenMSK/trajectory-prediction-for-KalmanPrediction-and-DeepLearning

import numpy as np

def GetScaleParam(current_dataset):
    min_position_x = 10000
    max_position_x = -10000
    min_position_y = 10000
    max_position_y = -10000
    
    # Generate datasetf
    x_ind = 3
    y_ind = 4
    
    #enumerate memberi nomor index untuk data, ind_directiory = index   raw_file_name = nama file 

        # file path yaitu /root/workspace/dataset/Apol/result_9060_6_frame.txt u/ masing2 file

    read = current_dataset
        #genfromtxt : data file dari txt disimpan ke read
        
        #untuk apol x_ind 3, y_ind 4
        #min(read[:, x_ind]) yaitu mencari nilai min dari kolom x_ind di file
    min_position_x = min(min_position_x, min(read[:, x_ind]))
    max_position_x = max(max_position_x, max(read[:, x_ind]))
    min_position_y = min(min_position_y, min(read[:, y_ind]))
    max_position_y = max(max_position_y, max(read[:, y_ind]))
    #mengembalikan nilai min dan max kolom x_ind, y_ind dari seluruh file dalam train atau val atau tes
    return min_position_x, max_position_x, min_position_y, max_position_y 

def normalize_data(current_dataset):
    x_ind = 3
    y_ind = 4

    min_position_x, max_position_x, min_position_y, max_position_y = GetScaleParam(current_dataset)
    print(min_position_x, max_position_x, min_position_y, max_position_y)

    current_dataset[:, x_ind] = (
                (current_dataset[:, x_ind] - min_position_x) / (max_position_x - min_position_x)
            ) * 2 - 1
    current_dataset[:, y_ind] = (
                (current_dataset[:, y_ind] - min_position_y) / (max_position_y - min_position_y)
            ) * 2 - 1

    return current_dataset

def GetCurrentData(idx, dataset):
    data_ind = idx
        
        # get current dataset (load data dalam file tempat idx berada)
    current_dataset = dataset
    current_dataset = normalize_data(current_dataset)

        #isi self.current_dataset adalah seluruh line dlm file yg mengandung idx
        
        #current_data berisi informasi sepanjang line(line tsb adalah idx) 
        #hanya 1 line
        #-> [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading]
    current_data = current_dataset[data_ind]

    return current_data, current_dataset

def get_history(agentId, frame, ref_agentId, current_dataset, historis_length):
    f_ind = 0
    id_ind = 1
    type_ind = 2
    x_ind = 3
    y_ind = 4
    f_interval = 1
    obs_length = historis_length
        # Based on the reference trajectory, get a relative trajectory
        #ref_track menyimpan data sepanjang line yang memiliki id 233 (sesuai idx)
    ref_track = current_dataset[current_dataset[:,id_ind]==agentId].astype(float)
        
        #dari ref_track cari frame 116 (sesuai idx) dan ambil data di elemen 3(posisi x) dan 4(posisi y)
    ref_pose = ref_track[ref_track[:,0]==frame][0, [x_ind, y_ind]]
        
        #agent_track isinya sama dg ref_track yaitu kumpulan data yg memiliki id 233 (sesuai idx)
    agent_track = current_dataset[current_dataset[:,id_ind]==agentId].astype(float)
        
        # for setting interval

        #np.argwhere madl fungsi untuk mengembalikan [baris kolom] dari matriks, index dimulai dr 0
        #mengatur interval data untuk histori
        # np.argwhere(agent_track[:, 0] == frame).item(0) -> agen dg id 233 di frame 116 ada di baris ke 30 (0:30) dlm agent_track
        #30 - (6 - 1)*1 = 15 < 0     false
        #obs_length akan tetap 6 jika letak agen dg id 233 frame 116 ada di >= 6 (0:6)
        #obs_length akan berubah menjadi letak agen dg id 233 frame 116 yang ditambah 1 jika letak agen dg id 233 frame 116 ada di < 6 (0:6)
    obs_length = int(np.argwhere(agent_track[:, 0] == frame).item(0)/f_interval + 1) if np.argwhere(agent_track[:, 0] == frame).item(0) - (obs_length-1)*(f_interval) < 0 else obs_length
        
        #nilai maksimum dr 0 dan 25
    start_idx = np.maximum(0, np.argwhere(agent_track[:, 0] == frame).item(0) - (obs_length-1)*f_interval)
        #30+1
    end_idx = np.argwhere(agent_track[:, 0] == frame).item(0) + 1
        
        # print("hist::::")
        # print("frame: ", agent_track[start_idx:end_idx:self.f_interval, 0])
        # frame:  [111. 112. 113. 114. 115. 116.]
        # agent_track[25:31:1 , 0]  -> 25 adalah line ke 25 (0:25)   -> 31 yang dimaksud adalah sampai menampilkan elemen di line ke 30 (0:30)
        # agent_track[start_idx:end_idx:f_interval,[x_ind, y_ind]] memberikan nilai posisi yg berada di frame sepanjang obs_length
        #
        # setekah dikurangi ref_pose maka menjadi posisi relatif terhadap frame 116
        #isi hist adalah posisi yang telah dinormalisasi milik id 233 frame start_idx:end_idx
        #array([[ 2.48405477e-04, -9.32572121e-05],
        #       [ 1.43962265e-04, -1.57371545e-04],
        #       [ 1.01620422e-04, -1.28228667e-04],
        #       [ 4.23418426e-05, -9.32572121e-05],
        #       [ 4.51646321e-05, -2.33143030e-05],
        #       [ 0.00000000e+00,  0.00000000e+00]])


    hist = agent_track[start_idx:end_idx:f_interval,[x_ind, y_ind]]# Get only relative positions [m]
        
        # mengecek frame sepanjang [start_idx:end_idx:f_interval, 0] sesuai
    reasonable_inds = np.where(agent_track[start_idx:end_idx:f_interval, 0] >= frame-f_interval*(obs_length-1))[0]
        # print(reasonable_inds)
 
        # jika frame yg telah tersimpan di hist tidak sesuai, maka akan di rewhrite berdasarkan reasonable_inds
    hist = hist[reasonable_inds]
        # print("HIST: ", hist)
    hist_mask = len(hist) #panjang baris histori
        
        #hanya masuk jika panjang hist kurang dr args.obs_length
        # jika baris histori kurang maka baris pertama histori akan di gandakan sejumlah baris yg kurang dan diletakkan diawal
    if len(hist) < obs_length:
            #np.full membuat matriks berukuran (obs_length,2) dengan semua elemen berisi hist[0] (baris ke 0 dr hist)
        tmp0 = np.full((obs_length,2), hist[0])
            #print(tmp0)
            # tmp0 = np.full((self.obs_length,2), 1e-6)
    
            #tmp0.shape[0] adalah menghitung ukuran dari terluar, 0 yang paling luar (isinya ada 6 baris) dan 1 yang paling dalam (pd tiap baris berisi 2 elemen)
        tmp0[tmp0.shape[0]-hist.shape[0]:,:] = hist
            #print(tmp0)
        return tmp0, ref_pose, hist_mask
        #yg dikembalikan tmp0 jika baris hist < obs_length
    return hist, ref_pose, hist_mask

def get_future(agentId, frame, current_dataset, future_length):
    f_ind = 0
    id_ind = 1
    type_ind = 2
    x_ind = 3
    y_ind = 4
    f_interval = 1
    pred_length = future_length

    agent_track = current_dataset[current_dataset[:,id_ind]==agentId].astype(float)
    ref_pose = agent_track[agent_track[:,0]==frame][0, [x_ind, y_ind]]
        
        #31
    start_idx = np.argwhere(agent_track[:, 0] == frame).item(0)+f_interval #t+1 frame
        #min dari 31 dan 30 + 10*1 + 1
    end_idx = np.minimum(len(agent_track), np.argwhere(agent_track[:, 0] == frame).item(0) + pred_length*f_interval + 1)#t+future frame
        
        # start_idx = np.argwhere(agent_track[:, 0] == frame).item(0)+1#t+1 frame
        # end_idx = np.minimum(len(agent_track), np.argwhere(agent_track[:, 0] == frame).item(0) + self.pred_length + 1)#t+future frame
        # print("fut::::")
        # print("frame: ", agent_track[start_idx:end_idx:self.f_interval, 0])

    fut = agent_track[start_idx:end_idx:f_interval,[x_ind, y_ind]]

    reasonable_inds = np.where(agent_track[start_idx:end_idx:f_interval, 0]<=frame+f_interval*pred_length)[0]
    fut = fut[reasonable_inds]
        # print("FUT: ", fut)
    return fut

def getData(idx, obs_length, pred_length, dataset):
    f_ind = 0 
    id_ind = 1
    type_ind = 2
    current_data, current_dataset = GetCurrentData(idx, dataset)
#dsId = dataset_path# dataset Id   ('/root/workspace/dataset/Apol/result_9061_1_frame.txt', 116, 233, 1)
    frame = current_data[f_ind].astype(int)    # frame in the dataset Id
    agentId = current_data[id_ind].astype(int)   # unique agentID in the dataset
    agentType = current_data[type_ind].astype(int)
# pose = self.current_data[[self.x_ind, self.y_ind, self.yaw_ind]]# [x, y, yaw]
    AgentInfo = (frame, agentId, agentType)

    hist, ref_pose, hist_mask = get_history(agentId, frame, agentId, current_dataset, obs_length)

    fut = get_future(agentId, frame, current_dataset, pred_length)
    fut_mask = len(fut)

    return hist, fut, ref_pose, AgentInfo