

# Image multi-task classfication

Easy-to-use/Easy-to-deploy/Easy-to-develop

<img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width = "300" height = "200" alt="图片名称" align=center> <img src="https://user-images.githubusercontent.com/9102141/87268895-3e0d0780-c4fe-11ea-849e-6140b7e0d4de.gif" width = "300" height = "200" alt="图片名称" align=center>


|      ***       |        |    example   |  
| :-----------------: | :---------:| :---------:|
|  models  |   (...等)       |  [1](./qdnet_classifier/models/)  |
|  metric  |   (Swish/ArcMarginProduct_subcenter/ArcFaceLossAdaptiveMargin/...)       |  [2](./qdnet/models/metric_strategy.py)  |
|  data aug  |   (rotate/flip/...、mixup/cutmix)         |  [3](./qdnet/dataaug/) | 
|  loss  |   (ce_loss/ce_smothing_loss/focal_loss/bce_loss/...)                     |  [4](./qdnet/loss/)    | 
|  deploy  |   (flask/grpc/BentoML等)                   |  [5](./serving/)       | 
|  onnx/trt |   ()                                      |  [6](./tools/)         | 


#

## train/test/deploy
0、Data format transform 
```
git clone https://github.com/MachineLP/PyTorch_image_classifier
pip install -r requirements.txt
cd PyTorch_image_classifier
python tools/data_preprocess_multi_task.py --data_dir "./data/data.csv" --n_splits 3 --output_dir "./data/train.csv" --random_state 2020
```

## resnet18
1、Modify configuration file

```
cp conf/resnet18.yaml conf/resnet18.yaml
vim conf/resnet18.yaml
```

2、Train: 

```
python train_multi_task.py --config_path conf/resnet18.yaml
```

3、Infer
```
    python infer_multi_task.py --config_path "conf/resnet18.yaml" --img_path "./data/img/0male/1_2.jpg" --fold "0"
    pre>>>>> [0] [0.6254628] [2] [0.8546583]
    python infer_multi_task.py --config_path "conf/resnet18.yaml" --img_path "./data/img/1female/2_5.jpg" --fold "1"
```


4、Models transform ( https://github.com/NVIDIA-AI-IOT/torch2trt ) ([Tensorrt installation guide on Ubuntu1804](./docs/Tensorrt_installation_guide_on_Ubuntu1804.md))

```
    onnx：python tools/pytorch_to_onnx_multi_task.py --config_path "conf/resnet18.yaml" --img_path "./data/img/0male/1_2.jpg" --batch_size 4 --fold 0 --save_path "lp.onnx"

    tensorrt：python tools/onnx_to_tensorrt_multi_task.py --config_path "conf/resnet18.yaml" --img_path "./data/img/0male/1_2.jpg" --batch_size 4 --fold 0 --save_path "lp_pp.onnx" --trt_save_path "lp.trt"
```


5、Deploying models
[serving](./serving/) 



#

#

#

#

#

#

#

#### ref
```
（1）https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
（2）https://github.com/BADBADBADBOY/pytorchOCR
（3）https://github.com/MachineLP/QDServing
（4）https://github.com/bentoml/BentoML
（5）mixup-cutmix:https://blog.csdn.net/u014365862/article/details/104216086
（7）focalloss:https://blog.csdn.net/u014365862/article/details/104216192
（8）https://blog.csdn.net/u014365862/article/details/106728375 / https://blog.csdn.net/u014365862/article/details/106728402 
```





