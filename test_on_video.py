# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QDNet infer
   Author :       machinelp
   Date :         2020-10-20
-------------------------------------------------
'''
import os
import time
import random
import sys 
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from qdnet.conf.config import load_yaml
from qdnet.optimizer.optimizer import GradualWarmupSchedulerV2
from qdnet.dataset.dataset import get_df, QDDataset
from qdnet.dataaug.dataaug import get_transforms
from qdnet.models.effnet import Effnet
from qdnet.models.resnest import Resnest
from qdnet.models.se_resnext import SeResnext
from qdnet.models.resnet import Resnet
from qdnet.conf.constant import Constant
from qdnet_classifier.classifier_multi_label import MultiLabelModel
from mtcnn import MTCNN
import onnxruntime



device = torch.device('cuda')

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--config_path', help='config file path')
parser.add_argument('--img_path', help='config file path')
parser.add_argument('--fold', help='config file path')
args = parser.parse_args()
config = load_yaml(args.config_path, args)
Sigmoid_fun = nn.Sigmoid()


class QDNetModel():

    def __init__(self, config, fold):

        if config["enet_type"] in Constant.RESNET_LIST:
            ModelClass = MultiLabelModel
        else:
            raise NotImplementedError()

        if config["eval"] == 'best':     
            model_file = os.path.join(config["model_dir"], f'best_fold{fold}.pth')
        if config["eval"] == 'final':    
            model_file = os.path.join(config["model_dir"], f'final_fold{fold}.pth')
        self.model = ModelClass(
            config["enet_type"],
            config["out_dim"],
            pretrained = config["pretrained"] )
        self.model = self.model.to(device)

        try:  # single GPU model_file
            self.model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        _, self.transforms_val = get_transforms(config["image_size"])  


    def predict(self, data):
        #if os.path.isfile(data):
        #    image = cv2.imread(data)
        #    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #else:
        image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        res = self.transforms_val(image=image)
        image = res['image'].astype(np.float32)

        image = image.transpose(2, 0, 1)
        data = torch.tensor([image]).float()
        probs = self.model( data.to(device) )
        return probs
    

'''
if __name__ == '__main__':

    qd_model = QDNetModel(config, args.fold)
    start_time = time.time()
    for i in range (10): 
        probs = qd_model.predict(args.img_path)
    print ("time>>>>", (time.time() - start_time)/10.0 )
    print ("pre>>>>>", Sigmoid_fun(probs))
'''

if __name__ == '__main__' :
 
    # Read video
    # video = cv2.VideoCapture("video/WeChatSight1395.mp4")
    video = cv2.VideoCapture(0)
    qd_model = QDNetModel(config, args.fold)
    mtcnn = MTCNN('./pb/mtcnn.pb')
    session = onnxruntime.InferenceSession("./v3.onnx", None)
    input_name = session.get_inputs()[0].name
 
    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read first frame.
    for i in range(10):
        ok, frame = video.read()
    
    h, w, c = frame.shape
    
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        img = frame
        h, w, c = img.shape
        bbox, scores, landmarks = mtcnn.detect(img)
        if len( bbox ) > 0:
            box = bbox[0]

            src_img = img[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
            img1 = cv2.resize(src_img, (112, 112))
            image_data = img1.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255
            output = session.run([], {input_name: image_data})[1]

            landmarks = output.reshape(-1, 2)
            landmarks[:, 0] = landmarks[:, 0] * src_img.shape[1]
            landmarks[:, 1] = landmarks[:, 1] * src_img.shape[0]
            


            # print (">>>>>", box)
            '''
            y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            h = y2-y1
            w = x2 -x1
            y1, x1, y2, x2 = int(y1-h*0.2), x1, int(y2+h*0.1), x2
            h = y2-y1
            w = x2 -x1
            if h>w:
                gap = (h-w)//2
                y1, x1, y2, x2 = y1, x1-gap, y2, x2+gap
            else:
                gap = (w-h)//2
                y1, x1, y2, x2 = y1-gap, x1, y2+gap, x2
            
            '''

            y1, x1, y2, x2 = int(box[0])+int(min(landmarks[:, 1])), int(box[1])+int(min(landmarks[:, 0])), int(box[0])+int(max(landmarks[:, 1])), int(box[1])+int(max(landmarks[:, 0]))

            print (">>>>>", y1, x1, y2, x2)
            
            h = y2-y1
            w = x2 -x1
            y1, x1, y2, x2 = int(y1-h*0.5), x1, int(y2+h*0.1), x2
            h = y2-y1
            w = x2 -x1
            if h>w:
                gap = (h-w)//2
                y1, x1, y2, x2 = y1, x1-gap, y2, x2+gap
            else:
                gap = (w-h)//2
                y1, x1, y2, x2 = y1-gap, x1, y2+gap, x2
            

            # y1, x1, y2, x2 = y1-50, x1-30, y2+20, x2+30
            src_img = img[ y1:y2, x1:x2, : ]
            # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite( "./test.jpg", src_img )

            probs = Sigmoid_fun(qd_model.predict(src_img))
            print (">>>>>", probs)
            cv2.putText(img, "Negative       :{}".format( probs[0][0].cpu().detach().numpy()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Normal         :{}".format( probs[0][1].cpu().detach().numpy()), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occlusion      :{}".format( probs[0][2].cpu().detach().numpy()), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occ-left-eye   :{}".format( probs[0][3].cpu().detach().numpy()), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occ-right-eye  :{}".format( probs[0][4].cpu().detach().numpy()), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occ-Nose       :{}".format( probs[0][5].cpu().detach().numpy()), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occ-mouth      :{}".format( probs[0][6].cpu().detach().numpy()), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occ-face-mask  :{}".format( probs[0][7].cpu().detach().numpy()), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occ-glasses    :{}".format( probs[0][8].cpu().detach().numpy()), (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Occ-sun-glasses:{}".format( probs[0][9].cpu().detach().numpy()), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


            for (x, y) in landmarks:
                cv2.circle(frame, (int(box[1])+int(x), int(box[0])+int(y)), 2, (0, 0, 255), -1)
            
            frame = cv2.rectangle( img, (x1, y1), (x2,y2), (255,0,0), 2 )
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

'''
python test_on_video.py --config_path "conf/mobilenetv3_small_multilabel.yaml" --img data/test_img/001.jpg --fold 0
'''