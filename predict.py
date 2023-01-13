'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要将预测结果保存成txt，可以利用open打开txt文件，使用write方法写入txt，可以参考一下txt_annotation.py文件。
'''
from PIL import Image
import importlib
import argparse

from classification import Classification



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="configs.base" 
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')
    parser.add_argument("--mode", type=str, default="dir_predict" , help="predict or dir_predict")  
    parser.add_argument("--dir_origin_path", type=str, 
                                        default="img/", 
                                        )  
    parser.add_argument("--dir_save_path", type=str, 
                                        default="img_out/", 
                                        )  


    conf = parser.parse_args() 
    opt = importlib.import_module(conf.config).get_opts(Train=False)

    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    
    d=vars(opt)

    #----------------------------------------------------------------------------------------------------------#
    dir_origin_path = opt.dir_origin_path
    dir_save_path   = opt.dir_save_path
    fps_image_path  = "img/fps.jpg"
    #-------------------------------------------------------------------------#

    classfication = Classification(root_dir=opt.out_path, 
                                    classes_path=opt.classes_path, 
                                    input_shape=opt.input_shape,
                                    backbone=opt.net)
    #-------------------------------------------------------------------------#                                

    mode = opt.mode
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                class_name = classfication.detect_image(image)
                print(class_name)

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = classfication.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
