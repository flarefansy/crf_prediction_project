from tensorflow.python import pywrap_tensorflow  
import os
import csv
import numpy as np
np.set_printoptions(threshold=np.inf)  
checkpoint_path = os.path.join('checkpointss_1', 'best_training')  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map = reader.get_variable_to_shape_map()  
checkpoint = []

for key in var_to_shape_map:  
    print("tensor_name: ", key)  
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names
    checkpoint.append(reader.get_tensor(key))

