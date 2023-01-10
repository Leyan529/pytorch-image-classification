import sys
sys.path.append(".")

import argparse, os, json
import torch
import torchvision as tv
from torch.utils.tensorboard import SummaryWriter
from utils.tools import init_logging
import importlib
from utils.helpers import get_data 
import numpy as np


def get_opts(Train=True):
    opt = argparse.Namespace()  

    #the train data, you need change.
    opt.data_root = '/home/leyan/DataSet/'
    # opt.data_root = "/home/zimdytsai/leyan/DataSet"
    # opt.data_root = 'D://WorkSpace//JupyterWorkSpace//DataSet//'


    opt.out_root = 'work_dirs/'
    opt.exp_name = 'SCUT'
    """
    [ icme, coco, voc, lane, cityscape ]
    """
    # get annotation file in current seting
    # importlib.import_module("annotation.{}".format(opt.exp_name)).get_annotation(opt.data_root) 

    opt.data_path, opt.class_names, opt.num_classes, annotation_path = \
            get_data(opt.data_root, opt.exp_name)

    #----------------------------------------------------#
    #   驗證集的劃分在train.py代碼里面進行
    #----------------------------------------------------#
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    val_split       = 0.1
    opt.num_val     = int(len(lines) * val_split)
    opt.num_train   = len(lines) - opt.num_val
    opt.lines = lines

    #############################################################################################
    #   phi             所使用到的yolov7的版本，本仓库一共提供两个：
    #                   l : 对应yolov7
    #                   x : 对应yolov7_x
    #############################################################################################    
    opt.net = 'resnet152'     # [unet, pspnet, segnet, fcn, deconvnet, fpn, deeplab_v3, deeplab_v3_plus, segformer]
    opt.model_path      = '' #coco
    opt.input_shape     = [299, 299] 
    opt.pretrained      = True
    opt.IM_SHAPE = (opt.input_shape[0], opt.input_shape[1], 3)    
    #------------------------------------------------------#
    opt.Freeze_Train    = True
    #----------------------------------------------------#
    #   凍結階段訓練參數
    #   此時模型的主幹被凍結了，特征提取網絡不發生改變
    #   占用的顯存較小，僅對網絡進行微調
    #----------------------------------------------------#
    opt.ngpu = 2
    opt.Init_Epoch          = 0
    opt.Freeze_Epoch        = 50 #50
    opt.Freeze_batch_size   = 32
    opt.Freeze_lr           = 1e-3
    #----------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特征提取網絡會發生改變
    #   占用的顯存較大，網絡所有的參數都會發生改變
    #----------------------------------------------------#
    opt.UnFreeze_Epoch  = 100 #100
    opt.Unfreeze_batch_size = 16
    opt.Unfreeze_lr         = 1e-4
    #------------------------------------------------------#
    #   是否進行凍結訓練，默認先凍結主幹訓練後解凍訓練。
    #------------------------------------------------------#   
    opt.loss_type      = "cce"         # [bce, cce, focal_loss]   
    #-------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    #-------------------------------------------------------------------#
    opt.batch_size = opt.Freeze_batch_size if opt.Freeze_Train else opt.Unfreeze_batch_size
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    opt.Init_lr             = 1e-2
    opt.Min_lr              = opt.Init_lr * 0.01
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    opt.lr_decay_type       = "cos"
    opt.weight_decay    = 5e-4
    opt.gamma           = 0.94
    opt.optimizer_type      = "sgd"
    opt.momentum            = 0.937
    #------------------------------------------------------#
    #   是否提早結束。
    #------------------------------------------------------#
    opt.Early_Stopping  = True
    #------------------------------------------------------#
    #   主幹特征提取網絡特征通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Freeze_Epoch為凍結訓練的世代
    #   UnFreeze_Epoch總訓練世代
    #   提示OOM或者顯存不足請調小Batch_size
    #------------------------------------------------------#
    opt.UnFreeze_flag = False
    #-------------------------------------------------------------------#
    #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
    #-------------------------------------------------------------------#
    opt.batch_size = opt.Freeze_batch_size if opt.Freeze_Train else opt.Unfreeze_batch_size
    opt.end_epoch = opt.Freeze_Epoch if opt.Freeze_Train else opt.UnFreeze_Epoch
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   內存較小的電腦可以設置為2或者0  
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #------------------------------------------------------#
    opt.num_workers         = 4
    opt.Cuda                = True
    opt.distributed         = True
    opt.sync_bn             = True
    opt.fp16                = True
    #############################################################################################
    opt.debug = 0
    ### Other ###
    opt.manual_seed = 704
    opt.log_batch_interval = 10
    opt.log_checkpoint = 10
    try:
        opt.local_rank  = int(os.environ["LOCAL_RANK"])
    except:
        opt.local_rank  = 0
    opt.ngpus_per_node  = torch.cuda.device_count()
    #############################################################################################
    opt.out_path = os.path.join(opt.out_root, "{}_{}".format(opt.exp_name, opt.net))
    if Train:
        opt.writer = SummaryWriter(log_dir=os.path.join(opt.out_path, "tensorboard"))
        init_logging(opt.local_rank, opt.out_path)    
 
    return opt

if __name__ == "__main__":    
    get_opts(Train=False)


