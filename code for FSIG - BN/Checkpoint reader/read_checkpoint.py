from tensorflow.python import pywrap_tensorflow  
import os
import csv
import numpy as np
import scipy.io as sio

np.set_printoptions(threshold=np.inf)  
checkpoint_path = os.path.join('checkpointss', 'best_training')  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map = reader.get_variable_to_shape_map()  
checkpoint = []

for key in var_to_shape_map:  
    print("tensor_name: ", key)  
   #print(reader.get_tensor(key)) # Remove this is you want to print only variable names
    checkpoint.append(reader.get_tensor(key))

sio.savemat('saveddata1.mat', {'biase_1st':checkpoint[0]})
sio.savemat('saveddata2.mat', {'scale_1st':checkpoint[1]})
sio.savemat('saveddata3.mat', {'weight_1st':checkpoint[2]})
sio.savemat('saveddata4.mat', {'mean_1st':checkpoint[3]})
sio.savemat('saveddata5.mat', {'shift_1st':checkpoint[4]})
sio.savemat('saveddata6.mat', {'var_1st':checkpoint[5]})
#

#sio.savemat('saveddata4.mat', {'biase_2st':checkpoint[4]})
#sio.savemat('saveddata5.mat', {'weight_out':checkpoint[5]})
#sio.savemat('saveddata6.mat', {'biase_out':checkpoint[6]})

#sio.savemat('saveddata1.mat', {'weight_1st':checkpoint[1]})
#sio.savemat('saveddata2.mat', {'biase_1st':checkpoint[0]})
#sio.savemat('saveddata3.mat', {'weight_2st':checkpoint[2]})
#sio.savemat('saveddata4.mat', {'biase_2st':checkpoint[4]})
#sio.savemat('saveddata5.mat', {'weight_out':checkpoint[5]})
#sio.savemat('saveddata6.mat', {'biase_out':checkpoint[6]})
#
#sio.savemat('saveddata7.mat', {'scale_1st':checkpoint[2]})
#sio.savemat('saveddata8.mat', {'shift_1st':checkpoint[3]})
#sio.savemat('saveddata7.mat', {'scale_1st':checkpoint[8]})
#sio.savemat('saveddata8.mat', {'shift_1st':checkpoint[3]})
#sio.savemat('saveddata9.mat', {'scale_2st':checkpoint[7]})
#sio.savemat('saveddata10.mat', {'shift_2st':checkpoint[9]})

#{'biase_1st':checkpoint[0]})#{'weight_2st':checkpoint[2]})#, {'biase_2st':checkpoint[3]}, {'weight_out':checkpoint[4]}, {'biase_out':checkpoint[5]})
