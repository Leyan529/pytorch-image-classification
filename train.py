import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import get_model_from_name, ModelType

from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes, weights_init
from utils.utils_fit import fit_one_epoch
from helps.choose_data import DataType, get_data

if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用Cuda
    #   沒有GPU可以設置成False
    #----------------------------------------------------#
    Cuda            = True
    #----------------------------------------------------#
    #   訓練自己的數據集的時候一定要注意修改classes_path
    #   修改成自己對應的種類的txt
    #----------------------------------------------------#
    root_path = "D:/WorkSpace/JupyterWorkSpace/DataSet/Image-Classification"
    data_dir, classes_path = get_data(root_path, DataType.SCUT)
    #----------------------------------------------------#
    #   輸入的圖片大小
    #----------------------------------------------------#
    # input_shape     = [224, 224]
    input_shape     = [299, 299]
    #----------------------------------------------------#
    #   所用模型種類：
    #   mobilenet、resnet50、vgg16、vit
    #
    #   在使用vit時學習率需要設置的小一些，否則不收斂
    #   可以將最下方的兩個lr分別設置成1e-4、1e-5
    #----------------------------------------------------#  
    backbone = ModelType.densenet       
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主幹網絡的預訓練權重，此處使用的是主幹的權重，因此是在模型構建的時候進行加載的。
    #   如果設置了model_path，則主幹的權值無需加載，pretrained的值無意義。
    #   如果不設置model_path，pretrained = True，此時僅加載主幹開始訓練。
    #   如果不設置model_path，pretrained = False，Freeze_Train = Fasle，此時從0開始訓練，且沒有凍結主幹的過程。
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   權值文件的下載請看README，可以通過網盤下載。模型的 預訓練權重 對不同數據集是通用的，因為特征是通用的。
    #   模型的 預訓練權重 比較重要的部分是 主幹特征提取網絡的權值部分，用於進行特征提取。
    #   預訓練權重對於99%的情況都必須要用，不用的話主幹部分的權值太過隨機，特征提取效果不明顯，網絡訓練的結果也不會好
    #
    #   如果訓練過程中存在中斷訓練的操作，可以將model_path設置成logs文件夾下的權值文件，將已經訓練了一部分的權值再次載入。
    #   同時修改下方的 凍結階段 或者 解凍階段 的參數，來保證模型epoch的連續性。
    #   
    #   當model_path = ''的時候不加載整個模型的權值。
    #
    #   此處使用的是整個模型的權重，因此是在train.py進行加載的，pretrain不影響此處的權值加載。
    #   如果想要讓模型從主幹的預訓練權值開始訓練，則設置model_path = ''，pretrain = True，此時僅加載主幹。
    #   如果想要讓模型從0開始訓練，則設置model_path = ''，pretrain = Fasle，此時從0開始訓練。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ""
    #------------------------------------------------------#
    #   是否進行凍結訓練，默認先凍結主幹訓練後解凍訓練。
    #------------------------------------------------------#
    Freeze_Train    = True
    #------------------------------------------------------#
    #   是否提早結束。
    #------------------------------------------------------#
    Early_Stopping  = False
    #------------------------------------------------------#
    #   獲得圖片路徑和標簽
    #------------------------------------------------------#    
    annotation_path = os.path.join(data_dir, "Classification/cls_train.txt")
    #------------------------------------------------------#
    #   進行訓練集和驗證集的劃分，默認使用10%的數據用於驗證
    #------------------------------------------------------#
    val_split       = 0.1
    #------------------------------------------------------#
    #   用於設置是否使用多線程讀取數據，0代表關閉多線程
    #   開啟後會加快數據讀取速度，但是會占用更多內存
    #   在IO為瓶頸的時候再開啟多線程，即GPU運算速度遠大於讀取圖片的速度。
    #------------------------------------------------------#
    num_workers     = 4
    #------------------------------------------------------#
    #   獲取classes
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    if backbone != "vit":
        model = get_model_from_name[backbone](num_classes = num_classes, pretrained = pretrained)
    else:
        model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_classes, pretrained = pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != "":
        #------------------------------------------------------#
        #   載入預訓練權重
        #------------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
        
    loss_history = LossHistory(os.path.join("logs", backbone), model_train, input_shape)
    #----------------------------------------------------#
    #   驗證集的劃分在train.py代碼里面進行
    #----------------------------------------------------#
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val     = int(len(lines) * val_split)
    num_train   = len(lines) - num_val

    #------------------------------------------------------#
    #   訓練分為兩個階段，分別是凍結階段和解凍階段。
    #   顯存不足與數據集大小無關，提示顯存不足請調小batch_size。
    #   受到BatchNorm層影響，batch_size最小為1。
    #   
    #   主幹特征提取網絡特征通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Freeze_Epoch為凍結訓練的世代
    #   Epoch為總訓練世代
    #   提示OOM或者顯存不足請調小batch_size
    #------------------------------------------------------#
    if True:
        #----------------------------------------------------#
        #   凍結階段訓練參數
        #   此時模型的主幹被凍結了，特征提取網絡不發生改變
        #   占用的顯存較小，僅對網絡進行微調
        #----------------------------------------------------#
        lr               = 1e-3
        # Batch_size     = 32
        Batch_size       = 2
        Init_Epoch       = 0
        max_Freeze_Epoch = 50

        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("數據集過小，無法進行訓練，請擴充數據集。")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        train_dataset   = DataGenerator(lines[:num_train], input_shape, random=True, train=True)
        val_dataset     = DataGenerator(lines[num_train:], input_shape, random=False, train=False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        #------------------------------------#
        #   凍結一定部分訓練
        #------------------------------------#
        if Freeze_Train:
            model.freeze_backbone()

        for epoch in range(Init_Epoch, max_Freeze_Epoch):
            next_UnFreeze_Epoch = epoch + 1
            if (Early_Stopping and loss_history.stopping): break
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, max_Freeze_Epoch, Cuda)
            lr_scheduler.step()
        print("End of Freeze Training")            
            

    if True:
        #----------------------------------------------------#
        #   解凍階段訓練參數
        #   此時模型的主幹不被凍結了，特征提取網絡會發生改變
        #   占用的顯存較大，網絡所有的參數都會發生改變
        #----------------------------------------------------#
        lr              = 1e-4
        Batch_size      = 16
        max_UnFreeze_Epoch  = 100

        epoch_step      = num_train // Batch_size
        epoch_step_val  = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("數據集過小，無法進行訓練，請擴充數據集。")

        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        train_dataset   = DataGenerator(lines[:num_train], input_shape, True)
        val_dataset     = DataGenerator(lines[num_train:], input_shape, False)
        gen             = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        gen_val         = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=detection_collate)
        #------------------------------------#
        #   解凍後訓練
        #------------------------------------#
        if Freeze_Train:
            model.Unfreeze_backbone()

        for epoch in range(next_UnFreeze_Epoch, max_UnFreeze_Epoch):
            if (Early_Stopping and loss_history.stopping): break
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, max_UnFreeze_Epoch, Cuda)
            lr_scheduler.step()
        print("End of UnFreeze Training")
