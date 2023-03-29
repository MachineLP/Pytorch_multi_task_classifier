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
from mtcnn import MTCNN

new_balck_list = []
balck_list = ['168.jpg','1188.jpg','1189.jpg','1650.jpg','1664.jpg','2003.jpg','3099.jpg','4081.jpg','4239.jpg','6425.jpg','6513.jpg','6536.jpg','7439.jpg','7804.jpg','7856.jpg','7905.jpg','8265.jpg','9388.jpg','9418.jpg','9793.jpg','9927.jpg','10418.jpg','10433.jpg','10770.jpg','11805.jpg','13869.jpg','13872.jpg','13873.jpg','13910.jpg','14239.jpg','14245.jpg','14321.jpg','14534.jpg','16403.jpg','16416.jpg','16438.jpg','17058.jpg','17683.jpg','18757.jpg','20679.jpg','20737.jpg','22313.jpg','22325.jpg','24755.jpg','24790.jpg','24814.jpg','24825.jpg','25080.jpg','25268.jpg','25921.jpg','26556.jpg','26565.jpg','26569.jpg','26582.jpg','27039.jpg','27462.jpg','27477.jpg','27538.jpg','27830.jpg','27909.jpg','28377.jpg','28438.jpg','28643.jpg','28661.jpg','28692.jpg','28693.jpg','28698.jpg','28738.jpg','28987.jpg','29344.jpg','29508.jpg','29652.jpg','29858.jpg','29856.jpg']
# glasses 
img_path = "./data/img/glasses"
all_img_name = os.listdir(img_path)
balck_list += all_img_name
img_path = "./data/img/sun_glasses"
all_img_name = os.listdir(img_path)
balck_list += all_img_name




mtcnn = MTCNN('./pb/mtcnn.pb')


'''
f = open("./data/img/occ_hand/data_label.txt")
lines = f.readlines()
for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]

    if img_name not in balck_list:
        
        occ_hand_label_list= ','.join( ['0'] + per_line[1:]+ ['0','0','0'] )
        img_occ_hand_label = occ_hand_label_list
        img_normal_label = ','.join(['1','0','0','0','0','0','0','0','0'])

        img_occ_hand_name = str('./data/img/occ_hand/' + img_name) 
        img_normal_name = str('./data/img/train_img/' + img_name) 

        img_occ_hand = cv2.imread( img_occ_hand_name )
        bbox, scores, landmarks = mtcnn.detect(img_occ_hand)
        if len(bbox) > 0 :
            box = bbox[0]
            src_img = img_occ_hand[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
            cv2.imwrite( str('./data/img/occ_hand_crop/' + img_name), src_img  )
            data_label_str = line
            with open(os.path.join('./data/img/occ_hand_crop/', 'data_label.txt') ,'a', encoding='utf-8') as f:    #设置文件对象
                f.write(data_label_str) 
        else:
            print ( "img_occ_hand_name:", img_occ_hand_name )
            # new_balck_list.append( img_name )

        img_normal = cv2.imread( img_normal_name )
        bbox, scores, landmarks = mtcnn.detect(img_normal)
        if len(bbox) > 0 :
            box = bbox[0]
            src_img = img_normal[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
            cv2.imwrite( str('./data/img/train_img_crop/' + img_name), src_img  )
        else:
            print ( "img_normal_name:", img_normal_name )
            # new_balck_list.append( img_name )





f = open("./data/img/occ_objects/data_label.txt")
lines = f.readlines()

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:

        occ_objects_label_list= ','.join( ['0'] + per_line[1:] + ['0','0','0'] )
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_objects_label = occ_objects_label_list
        img_occ_objects_name = str('./data/img/occ_objects/' + img_name) 

        img_occ_objects = cv2.imread( img_occ_objects_name )
        bbox, scores, landmarks = mtcnn.detect(img_occ_objects)
        if len(bbox) > 0 :
            box = bbox[0]
            src_img = img_occ_objects[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
            cv2.imwrite( str('./data/img/occ_objects_crop/' + img_name), src_img  )
            data_label_str = line
            with open(os.path.join('./data/img/occ_objects_crop/', 'data_label.txt') ,'a', encoding='utf-8') as f:    #设置文件对象
                f.write(data_label_str) 
        else:
            print ( "img_occ_objects_name:", img_occ_objects_name )
            # new_balck_list.append( img_name )

        



f = open("./data/img/occ_others/data_label.txt")
lines = f.readlines()

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:

        occ_others_label_list= ','.join( ['0'] + per_line[1:] + ['0','0','0'] )
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_others_label = occ_others_label_list
        img_occ_others_name = str('./data/img/occ_others/' + img_name) 
        
        img_occ_others = cv2.imread( img_occ_others_name )
        bbox, scores, landmarks = mtcnn.detect(img_occ_others)
        if len(bbox) > 0 :
            box = bbox[0]
            src_img = img_occ_others[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
            cv2.imwrite( str('./data/img/occ_others_crop/' + img_name), src_img  )
            data_label_str = line
            with open(os.path.join('./data/img/occ_others_crop/', 'data_label.txt') ,'a', encoding='utf-8') as f:    #设置文件对象
                f.write(data_label_str) 
        else:
            print ( "img_occ_others_name:", img_occ_others_name )
            # new_balck_list.append( img_name )





f = open("./data/img/occ_hand_normal/data_label.txt")
lines = f.readlines()
img_occ_hand_normal_name_list = []
img_occ_hand_normal_label_list = []

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:

        occ_hand_normal_label_list= ','.join( ['0'] + per_line[1:] + ['0','0','0'] )
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_hand_normal_label = occ_hand_normal_label_list
        img_occ_hand_normal_name = str('./data/img/occ_hand_normal/' + img_name) 
        
        img_occ_hand_normal = cv2.imread( img_occ_hand_normal_name )
        bbox, scores, landmarks = mtcnn.detect(img_occ_hand_normal)
        if len(bbox) > 0 :
            box = bbox[0]
            src_img = img_occ_hand_normal[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
            cv2.imwrite( str('./data/img/occ_hand_normal_crop/' + img_name), src_img  )
            data_label_str = line
            with open(os.path.join('./data/img/occ_hand_normal_crop/', 'data_label.txt') ,'a', encoding='utf-8') as f:    #设置文件对象
                f.write(data_label_str) 
        else:
            print ( "img_occ_hand_normal_name:", img_occ_hand_normal_name )
            # new_balck_list.append( img_name )
'''



# face mask
img_path = "./data/img/face_mask"
all_img_name = os.listdir(img_path)

for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)

    img_face_mask = cv2.imread( per_img_path )
    bbox, scores, landmarks = mtcnn.detect(img_face_mask)
    if len(bbox) > 0 :
        box = bbox[0]
        src_img = img_face_mask[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
        cv2.imwrite( str('./data/img/face_mask_crop/' + per_img_name), src_img  )
    else:
        print ( "img_face_mask_name:", per_img_path )
        new_balck_list.append( per_img_name )


# glasses 
img_path = "./data/img/glasses"
all_img_name = os.listdir(img_path)

for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_glasses = cv2.imread( per_img_path )
    bbox, scores, landmarks = mtcnn.detect(img_glasses)
    if len(bbox) > 0 :
        box = bbox[0]
        src_img = img_glasses[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
        cv2.imwrite( str('./data/img/glasses_crop/' + per_img_name), src_img  )
    else:
        print ( "img_glasses_name:", per_img_path )
        new_balck_list.append( per_img_name )


# sun glasses 
img_path = "./data/img/sun_glasses"
all_img_name = os.listdir(img_path)

for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_sun_glasses = cv2.imread( per_img_path )
    bbox, scores, landmarks = mtcnn.detect(img_sun_glasses)
    if len(bbox) > 0 :
        box = bbox[0]
        src_img = img_sun_glasses[ int(box[0]):int(box[2]), int(box[1]):int(box[3]), : ]
        cv2.imwrite( str('./data/img/sun_glasses_crop/' + per_img_name), src_img  )
    else:
        print ( "img_sun_glasses_name:", per_img_path )
        new_balck_list.append( per_img_name )


print ( list(set(new_balck_list)) )



