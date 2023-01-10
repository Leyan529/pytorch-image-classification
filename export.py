import argparse
import sys
import time
import warnings
import colorsys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import copy

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn.functional as F

import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import datetime
import logging

import models
import importlib
import onnxruntime as ort
import numpy as np
import cv2
from glob import glob
import os
from PIL import ImageDraw, ImageFont, Image
from scipy.special import softmax

def select_device(net, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'{net.upper()} 🚀 torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    print(s)
    return torch.device('cuda:0' if cuda else 'cpu')

def resize_image(image, size):
    ih, iw, _  = image.shape
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    new_image       = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # new_image = cv2.rectangle(new_image, ((w-nw)//2,0), (nw - ((w-nw)//2),nh), (128,128,128), 15)

    # cv2.imshow("gray bound", new_image)
    # cv2.waitKey(0)
    return new_image, nw, nh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # error
    # parser.add_argument('--config', type=str, default="configs.deconv_base" ,help = 'Path to config .opt file. ')      
    parser.add_argument('--config', type=str, default="configs.base" ,help = 'Path to config .opt file. ')

    parser.add_argument('--weights', type=str, default='best_epoch_weights.pth', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', default=True, action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch', action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--blend', type=bool, default=True, help='iou threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', default=True, action='store_true', help='simplify onnx model')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')

    opt = parser.parse_args()
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    conf = importlib.import_module(opt.config).get_opts(Train=False)
    for key, value in vars(conf).items():
        setattr(opt, key, value)
    opt.weights = os.path.join(opt.out_path, opt.weights)    

    # print(opt)
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.net, opt.device)

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.input_shape).to(device)  # image size(1,3,320,192) iDetection

    print("Load model.")
    model = models.get_model(opt.net, opt.input_shape, pretrained=opt.pretrained, output_size=opt.num_classes)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    print("Load model done.") 

    y = model(img)  # dry run

    if True:
        # ONNX export
        try:
            import onnx

            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = opt.weights.replace('.pth', '.onnx')  # filename
            model.eval()
            output_names = ['classes', 'boxes'] if y is None else ['output']

            dynamic_axes = None
            if opt.dynamic:
                dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                'output': {0: 'batch', 2: 'y', 3: 'x'}}            

            input_names = ['images']
            torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=input_names,
            # torch.onnx.export(model, img, f, verbose=False, opset_version=14, input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=dynamic_axes)

            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model

            graph = onnx.helper.printable_graph(onnx_model.graph)
            # print(graph)  # print a human readable model         
            # onnx_graph_path = opt.weights.replace(".pth", ".txt")
            # with open(onnx_graph_path, "w", encoding="utf-8") as f:
            #     f.write(graph)
            

            if opt.simplify:
                try:
                    import onnxsim

                    print('\nStarting to simplify ONNX...')
                    onnx_model, check = onnxsim.simplify(onnx_model)
                    assert check, 'assert check failed'
                except Exception as e:
                    print(f'Simplifier failure: {e}')

            # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
            f = opt.weights.replace('.pth', '_simp.onnx')  # filename
            onnx.save(onnx_model, f)
            print('ONNX export success, saved as %s' % f)


        except Exception as e:
            print('ONNX export failure: %s' % e)
            exit(-1)

        # Finish
        print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))


    # from det_model.yolov5.utils.utils_bbox import DecodeBox
    # f = os.path.join(opt.out_path, "best_epoch_weights_simp.onnx")
    f = os.path.join(opt.out_path, "best_epoch_weights.onnx")
    ort_session = ort.InferenceSession(f)  

    # Test forward with onnx session (test image) 
    video_path      = os.path.join(opt.data_path, "Drive-View-Kaohsiung-Taiwan.mp4")
    capture = cv2.VideoCapture(video_path)

    fourcc  = cv2.VideoWriter_fourcc(*'XVID')
    size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ref, frame = capture.read()

    fps = 0.0
    drawline = False
    # post = Post(opt)

    for frame in glob("extra/*.jpg") :
        frame = cv2.imread(frame)

    # while(True):
    #     t1 = time.time()
    #     # 讀取某一幀
    #     ref, frame = capture.read()
    #     if not ref:
    #         break
        t1 = time.time()
        #---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        #---------------------------------------------------#
        old_img     = copy.deepcopy(frame)
        orininal_h  = np.array(frame).shape[0]
        orininal_w  = np.array(frame).shape[1]
        #---------------------------------------------------#
        image_shape = np.array(np.shape(frame)[0:2])          

        # new_image       = cv2.resize(frame, opt.input_shape, interpolation=cv2.INTER_CUBIC)
        new_image, nw, nh  = resize_image(frame, (opt.input_shape[1], opt.input_shape[0]))
        new_image       = np.expand_dims(np.transpose(np.array(new_image, dtype=np.float32)/255, (2, 0, 1)),0)

        outputs = ort_session.run(
            None, 
            {"images": new_image
             },
        )

        preds   = softmax(outputs[0])
       

        #---------------------------------------------------#
        #   获得所属种类
        #---------------------------------------------------#
        class_name  = opt.class_names[np.argmax(preds)]
        probability = np.max(preds)

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("%s= %.2f"%(class_name, probability))
        frame = cv2.putText(frame, "%s= %.2f"%(class_name, probability), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        cv2.waitKey(0) 
        # if c==27:
        #     capture.release()
        #     break