#-*- coding:utf-8 _*-
import os
import sys
sys.path.append('./')
import yaml
import math
import argparse
import torch.nn as nn
import torch
import cv2
import numpy as np
import onnx
import time
import onnxruntime
from PIL import Image
from qdnet.dataaug.dataaug import get_transforms
from qdnet.conf.config import load_yaml

from qdnet.conf.config import load_yaml
from qdnet.optimizer.optimizer import GradualWarmupSchedulerV2
from qdnet.dataset.dataset import get_df, QDDataset
from qdnet.dataaug.dataaug import get_transforms
from qdnet.models.effnet import Effnet
from qdnet.models.resnest import Resnest
from qdnet.models.se_resnext import SeResnext
from qdnet.loss.loss import Loss
from qdnet.conf.constant import Constant
from qdnet_classifier.classifier_multi_label import MultiLabelModel
parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--img_path', nargs='?', type=str, default=None)
parser.add_argument('--config_path', help='config file path')
parser.add_argument('--batch_size', nargs='?', type=int, default=None)
parser.add_argument('--fold', help='config file path', type=int)
parser.add_argument('--save_path', help='config file path', type=str)
args = parser.parse_args()
config = load_yaml(args.config_path, args)


if config["enet_type"] in Constant.RESNET_LIST:
         ModelClass = MultiLabelModel
else:
    raise NotImplementedError()

model = ModelClass(
        config["enet_type"],
        config["out_dim"],
        pretrained = config["pretrained"] )
device = torch.device('cpu')
model = model.to(device)


def gen_onnx(args):


    if config["eval"] == 'best':
        model_file = os.path.join(config["model_dir"], f'best_fold{args.fold}.pth')
    if config["eval"] == 'final':
        model_file = os.path.join(config["model_dir"], f'final_fold{args.fold}.pth')


    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')), strict=True)

    model.eval()

    print('load model ok.....')


    img = cv2.imread(args.img_path)
    transforms_train, transforms_val = get_transforms(config["image_size"])
    # img1 = transforms.ToTensor()(img1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = transforms_val(image=img)
    img1 = res['image'].astype(np.float32)
    img1 = img1.transpose(2, 0, 1)
    img1 = torch.tensor([img1]).float()

    s = time.time()
    with torch.no_grad():
        out = model(img1.to(device))
        probs = out
        print ("probs>>>>>",probs)

    print('cost time:',time.time()-s)

    output_onnx = args.save_path
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    # output_names = ["hm" , "wh"  , "reg"]
    output_names = ["out"]
    dynamic_axes = {'input': {0: 'batch'}, 'out': {0: 'batch'}}
    inputs = torch.randn(args.batch_size, 3,128,128).cpu()
    '''
    export_type = torch.onnx.OperatorExportTypes.ONNX
    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,do_constant_folding=False,keep_initializers_as_inputs=True,
                                   input_names=input_names, output_names=output_names, operator_export_type=export_type, dynamic_axes=dynamic_axes)
    '''
    # torch.onnx.export(model, inputs, output_onnx, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    # torch.onnx.export(model, inputs, output_onnx, verbose=False, export_params=True, training=False, opset_version=10, example_outputs=probs, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    torch.onnx.export(model, inputs, output_onnx)
    
    onnx_path = args.save_path
    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    image = img1.cpu().detach().numpy()
    image = image.astype(np.float32)
    print (">>>>>", image.shape)
    s = time.time()
    preds = session.run([], {input_name: image})
    print ("preds>>>>>",preds)
    preds = preds[0]
    print('cost time:', time.time()-s)
    if isinstance(preds,dict):
        preds = preds['f_score']

    cv2.imwrite('./onnx/onnx_output.jpg',preds[0,0]*255)

    print('error_distance:',np.abs((out.cpu().detach().numpy()-preds)).mean())
    

if __name__ == "__main__":
    gen_onnx(args)
    os.system("pip install onnx-simplifier")
    os.system("python -m onnxsim lp.onnx lp_pp.onnx")

# python tools/pytorch_to_onnx_muliti_label.py --config_path "conf/tf_mobilenetv3_small_minimal_100_multilabel.yaml" --img_path "./data/test_img/001.jpg" --batch_size 4 --fold 0 --save_path "lp.onnx"