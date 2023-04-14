# ImageClassification
My Frame work for ImageClassification 
## Overview
I organizize the object detection algorithms proposed in recent years, and focused on **`AOI` and `SCUT` and `HandWrite`** Dataset.
This frame work also include **`EarlyStopping mechanism`**.


## Datasets:

I used 3 different datases: **`AOI`, `SCUT` and `HandWrite`** . Statistics of datasets I used for experiments is shown below

- **AOI**:
  This topic takes flexible electronic displays as the inspection target, and hopes to interpret the classification of defects through data science to improve the effectiveness of AOI.

    Download the classification images and organize folder from [AOI](https://aidea-web.tw/topic/252eb73e-78d0-4024-8937-40ed20187fd8). Make sure to put the files as the following structure:

  The image data provided in this topic includes 6 categories (normal category + 5 defect categories).

  ```
    AOI
    ├── train_images
    │   ├── 0
    │   ├── 1
    │   ├── 2  
    │   ├── 3 
    │   ├── 4 
    │   ├── 5 
    │     
    │── test_images
        ├── 0 (Default)
  ```
  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhksLwPvSDAYFucVb?e=ApCV9w)
  
- **SCUT-FBP5500**:
  A diverse benchmark database (Size = 172MB) for multi-paradigm facial beauty prediction is now released by Human Computer Intelligent Interaction Lab of South China University of Technology.

  Download the classification images and organize folder from [SCUT](https://drive.google.com/open?id=1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf). Make sure to put the files as the following structure:

  The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties (male/female, Asian/Caucasian, ages) and diverse labels (facial landmarks, beauty scores in 5 scales, beauty score distribution).
  I use the **`round of mean beauty score`** to train the classification model.

  ```
    SCUT-FBP5500_v2
    ├── train_images
    │   ├── 0
    │   ├── 1
    │   ├── 2  
    │   ├── 3 
    │   ├── 4 
    │     
    │── test_images
        ├── 0 (Default)
  ```
  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhkrExp6Hf37lMkOb?e=zoOMse)

- **HandWrite**:
  In the banking industry, there are various important handwritten documents that need to be manually inputted, which requires 21 person-hours per day. Therefore, we are currently looking for experts in image recognition who can use deep learning to automatically identify text in images and greatly reduce the cost of manual processing of repetitive tasks. The Yuanta AI Open Challenge 2021 Summer Games provides handwritten Chinese character image files for contestants to recognize text through CV algorithms and have the opportunity to win a high prize. The competition is divided into two stages: model training and online battles - model accuracy [competition](https://tbrain.trendmicro.com.tw/Competitions/Details/14) (test: 2021/05/24 - 2021/05/26, formal: 2021/06/15 - 2021/06/18).
  
  ![image](https://user-images.githubusercontent.com/24097516/231931129-f91e9682-01b0-4a8f-9ad8-c2f267aef6f1.png)

  Processed File: [download link](https://1drv.ms/u/s!AvbkzP-JBXPAhkyKEl4FXz-FiKRi?e=WcSLNX)

## Methods
- **mobilenet**
- **resnet50**
- **vgg16**
- **vit**
- **googlenet**
- **shufflenet**
- **inception**
- **squeezenet**
- **efficientnet**
- **densenet**
- **alexnet**
#### Pretrain-Weights [download link](https://1drv.ms/f/s!AvbkzP-JBXPAhXedeJneNFRtJAM5?e=OOWkfm)

## Prerequisites
* **Windows 10**
* **CUDA 10.1 (lower versions may work but were not tested)**
* **NVIDIA GPU 1660 + CuDNN v7.3**
* **python 3.6.9**
* **pytorch 1.10**
* **opencv (cv2)**
* **numpy**
* **torchvision 0.4**
* **torchsummary**
* **dlib==19.21**

## Requirenents

```python
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
### 1. Prepare the dataset
* **Create your own `dataset_annotation.py` then create `cls_train.txt` and `cls_test.txt` .** 
* **Prepare pretrain download weight to `model_data` .** 
* **For `SCUT` I prepare code to predict face in `predict.py`, which need prepare face_landmarks and dlib library to make `frontal_face_detector`, these data are put in extra**.
* **`shape_predictor_68_face_landmarks.dat` download from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)**.

### 2. Train (Freeze backbone + UnFreeze backbone) 
* setup your `root_path` and choose `DataType`
```python
python train.py
```

### 3. Evaluate  (eval_top1 / eval_top5) 
* setup your `root_path` and choose `DataType` in `eval_topn.py`
* setup your `model_path` and `classes_path` in `classification.py`
* evaluate file will load `classification.py` configuration
```python
python eval_top1.py
python eval_top5.py
```

### 4. predicy  
* setup your `model_path` and `classes_path` in `classification.py`
* evaluate file will load `classification.py` configuration
```python
python predict.py
```

### 5. export
* Can switch your saved model export to ONNX format
```python
python export.py --config "configs.yolact_base"
```
## Demo
![hZTSthw](https://user-images.githubusercontent.com/24097516/213659867-31b8e574-0169-4fba-8b12-35e1c6295e51.jpeg)

## Reference
- dlib-models : https://github.com/davisking/dlib-models
- Transfer Learning - Fine tune : https://hackmd.io/@lido2370/HyLTOlSn4?type=view
- ImageFolder : https://blog.csdn.net/TH_NUM/article/details/80877435
- pytorch-summary : https://github.com/sksq96/pytorch-summary
- classification-pytorch: https://github.com/bubbliiiing/classification-pytorch
