import os
import numpy as np


class DataType:
    EdgeAOI   = 0
    HandWrite  = 1
    SCUT  = 2 

def get_data(root_path, dataType):
    #   數據集路徑
    #   訓練自己的數據集必須要修改的
    #------------------------------#  
    if dataType == DataType.EdgeAOI:
        data_dir = os.path.join(root_path, "EdgeAOI/")
        classes_path    = 'model_data/EdgeAOI_classes.txt' 

    elif dataType == DataType.HandWrite:
        data_dir = os.path.join(root_path, "HandWrite/")
        classes_path    = 'model_data/HandWrite_classes.txt' 

    elif dataType == DataType.SCUT:
        data_dir = os.path.join(root_path, "SCUT-FBP5500_v2/")
        classes_path    = 'model_data/SCUT_classes.txt' 

    return data_dir, classes_path