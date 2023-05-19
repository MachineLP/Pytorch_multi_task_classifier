from pathlib import Path
import os
import cv2
from unicodedata import category
from pandas.core.frame import DataFrame
import numpy as np

# ['Negative', 'Normal', 'Occlusion', 'Occ-left-eye', 'Occ-right-eye', 'Occ-Nose', 'Occ-mouth', 'Occ-face-mask', 'Occ-glasses', 'Occ-sun-glasses']

balck_list = ['168.jpg','1188.jpg','all_img_name1189.jpg','1650.jpg','1664.jpg','2003.jpg','3099.jpg','4081.jpg','4239.jpg','6425.jpg','6513.jpg','6536.jpg','7439.jpg','7804.jpg','7856.jpg','7905.jpg','8265.jpg','9388.jpg','9418.jpg','9793.jpg','9927.jpg','10418.jpg','10433.jpg','10770.jpg','11805.jpg','13869.jpg','13872.jpg','13873.jpg','13910.jpg','14239.jpg','14245.jpg','14321.jpg','14534.jpg','16403.jpg','16416.jpg','16438.jpg','17058.jpg','17683.jpg','18757.jpg','20679.jpg','20737.jpg','22313.jpg','22325.jpg','24755.jpg','24790.jpg','24814.jpg','24825.jpg','25080.jpg','25268.jpg','25921.jpg','26556.jpg','26565.jpg','26569.jpg','26582.jpg','27039.jpg','27462.jpg','27477.jpg','27538.jpg','27830.jpg','27909.jpg','28377.jpg','28438.jpg','28643.jpg','28661.jpg','28692.jpg','28693.jpg','28698.jpg','28738.jpg','28987.jpg','29344.jpg','29508.jpg','29652.jpg','29858.jpg','29856.jpg']
# glasses 
img_path = "./data/img/glasses"
all_img_name = os.listdir(img_path)
balck_list += all_img_name
img_path = "./data/img/sun_glasses"
all_img_name = os.listdir(img_path)
balck_list += all_img_name

f = open("./data/img/occ_hand/data_label.txt")
lines = f.readlines()
img_occ_hand_name_list = []
img_normal_name_list = []
img_occ_hand_label_list = []
img_normal_label_list = []
fold_list = []
for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:
        
        occ_hand_label_list= ['0','0'] + per_line[1:]+ ['0','0','0'] 
        occ_hand_label_list[2] = '1'
        img_occ_hand_label = ','.join( occ_hand_label_list )
        img_normal_label = ','.join(['0','1','0','0','0','0','0','0','0','0'])

        img_occ_hand_name_list.append( str('./data/img/occ_hand/' + img_name) )
        img_normal_name_list.append( str('./data/img/train_img/' + img_name) )
        img_occ_hand_label_list.append( str(img_occ_hand_label) )
        img_normal_label_list.append( str(img_normal_label) )
        fold_list.append( 0 )
        fold_list.append( 0 )
'''
black_list3 = ['16500.jpg', '9139.jpg', '4585.jpg', '10106.jpg', '28372.jpg', '18705.jpg', '10268.jpg', '24892.jpg', '6290.jpg', '4907.jpg', '25607.jpg', 'data_label.txt', '28380.jpg', '8054.jpg', '24877.jpg', '17260.jpg', '12993.jpg', '21367.jpg', '5565.jpg', '5028.jpg', '5339.jpg', '5468.jpg', '19694.jpg', '21574.jpg', '11909.jpg', '5234.jpg', '5211.jpg', '29882.jpg', '5624.jpg', '22260.jpg', '22311.jpg', '24538.jpg', '28626.jpg', '4831.jpg', '7749.jpg', '936.jpg', '7650.jpg', '20753.jpg', '21418.jpg', '2979.jpg', '3705.jpg', '19006.jpg', '12580.jpg', '4977.jpg', '10324.jpg', '25148.jpg', '25224.jpg']
f = open("./data/img/occ_hand_crop/data_label.txt")
lines = f.readlines()
for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list+black_list3:
        
        occ_hand_label_list= ','.join( ['0','0'] + per_line[1:]+ ['0','0','0'] )
        img_occ_hand_label = occ_hand_label_list
        img_normal_label = ','.join(['0','1','0','0','0','0','0','0','0','0'])
        try:
            #img = cv2.imread( str('./data/img/train_img_crop/' + img_name) )
            #if img is None:
            #    print (img_name)
            img_occ_hand_name_list.append( str('./data/img/occ_hand_crop/' + img_name) )
            img_normal_name_list.append( str('./data/img/train_img_crop/' + img_name) )
            img_occ_hand_label_list.append( str(img_occ_hand_label) )
            img_normal_label_list.append( str(img_normal_label) )
            fold_list.append( 0 )
            fold_list.append( 0 )
        except:
            print (img_name)
            continue
'''     


f = open("./data/img/occ_objects/data_label.txt")
lines = f.readlines()
img_occ_objects_name_list = []
img_occ_objects_label_list = []

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:

        occ_objects_label_list = ['0','0'] + per_line[1:] + ['0','0','0']
        occ_objects_label_list[2] = '1'
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_objects_label = ','.join( occ_objects_label_list )
        img_occ_objects_name_list.append( str('./data/img/occ_objects/' + img_name) )
        img_occ_objects_label_list.append( str(img_occ_objects_label) )
        fold_list.append( 0 )

'''
f = open("./data/img/occ_objects_crop/data_label.txt")
lines = f.readlines()

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list+black_list3:

        occ_objects_label_list= ','.join( ['0','0'] + per_line[1:] + ['0','0','0'] )
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_objects_label = occ_objects_label_list
        img_occ_objects_name_list.append( str('./data/img/occ_objects_crop/' + img_name) )
        img_occ_objects_label_list.append( str(img_occ_objects_label) )
        fold_list.append( 0 )
'''

f = open("./data/img/occ_others/data_label.txt")
lines = f.readlines()
img_occ_others_name_list = []
img_occ_others_label_list = []

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:

        occ_others_label_list= ['0','0'] + per_line[1:] + ['0','0','0']
        occ_others_label_list[2] = '1'
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_others_label = ','.join( occ_others_label_list )
        img_occ_others_name_list.append( str('./data/img/occ_others/' + img_name) )
        img_occ_others_label_list.append( str(img_occ_others_label) )
        fold_list.append( 0 )


f = open("./data/img/occ_others2/data_label.txt")
lines = f.readlines()

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:

        occ_others_label_list= ['0','0'] + per_line[1:] + ['0','0','0']
        occ_others_label_list[2] = '1'
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_others_label = ','.join( occ_others_label_list )
        img_occ_others_name_list.append( str('./data/img/occ_others2/' + img_name) )
        img_occ_others_label_list.append( str(img_occ_others_label) )
        fold_list.append( 0 )

'''
f = open("./data/img/occ_others_crop/data_label.txt")
lines = f.readlines()

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list+black_list3:

        occ_others_label_list= ','.join( ['0','0'] + per_line[1:] + ['0','0','0'] )
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_others_label = occ_others_label_list
        img_occ_others_name_list.append( str('./data/img/occ_others_crop/' + img_name) )
        img_occ_others_label_list.append( str(img_occ_others_label) )
        fold_list.append( 0 )
'''

f = open("./data/img/occ_hand_normal/data_label.txt")
lines = f.readlines()
img_occ_hand_normal_name_list = []
img_occ_hand_normal_label_list = []

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list:

        occ_hand_normal_label_list= ['0','0'] + per_line[1:] + ['0','0','0']
        occ_hand_normal_label_list[2] = '1'
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_hand_normal_label = ','.join( occ_hand_normal_label_list )
        img_occ_hand_normal_name_list.append( str('./data/img/occ_hand_normal/' + img_name) )
        img_occ_hand_normal_label_list.append( str(img_occ_hand_normal_label) )
        fold_list.append( 0 )

'''
f = open("./data/img/occ_hand_normal_crop/data_label.txt")
lines = f.readlines()

for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name not in balck_list+black_list3:

        occ_hand_normal_label_list= ','.join( ['0','0'] + per_line[1:] + ['0','0','0'] )
        # occ_label_list = [ int(i) for i in per_line[1:] ]
        img_occ_hand_normal_label = occ_hand_normal_label_list
        img_occ_hand_normal_name_list.append( str('./data/img/occ_hand_normal_crop/' + img_name) )
        img_occ_hand_normal_label_list.append( str(img_occ_hand_normal_label) )
        fold_list.append( 0 )

black_list2 = black_list3 + ['22776.jpg', '11617.jpg', '1_0_1.jpg', '22909.jpg', '19485.jpg', '16373.jpg', 'jdasgjkadsgjaddadg.png', '0_0_006vBMIgjw1fabb8ghpxrj30im0cgjt5.jpg', '16438.jpg', '8853.jpg', '26299.jpg', '20754.jpg', 'OK-mask_0109.jpg', '6557.jpg', '0_0_5.jpg', '15343.jpg', '6010.jpg', '20055.jpg']
'''

# face mask
img_path = "./data/img/face_mask"
all_img_name = os.listdir(img_path)

img_face_mask_name_list = []
img_face_mask_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_face_mask_label = ','.join(['0','0','1','0','0','1','1','1','0','0'])

    img_face_mask_name_list.append( per_img_path )
    img_face_mask_label_list.append( img_face_mask_label )
    fold_list.append( 0 )

'''
img_path = "./data/img/face_mask_crop"
all_img_name = os.listdir(img_path)

for per_img_name in all_img_name:
    if per_img_name not in black_list2:
        per_img_path = os.path.join(img_path, per_img_name)
        img_face_mask_label = ','.join(['0','0','1','0','0','1','1','1','0','0'])

        img_face_mask_name_list.append( per_img_path )
        img_face_mask_label_list.append( img_face_mask_label )
        fold_list.append( 0 )
'''

# glasses 
img_path = "./data/img/glasses"
all_img_name = os.listdir(img_path)

img_glasses_name_list = []
img_glasses_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_glasses_label = ','.join(['0','0','1','0','0','0','0','0','1','0'])

    img_glasses_name_list.append( per_img_path )
    img_glasses_label_list.append( img_glasses_label )
    fold_list.append( 0 )

'''
img_path = "./data/img/glasses_crop"
all_img_name = os.listdir(img_path)

for per_img_name in all_img_name:

    if per_img_name not in black_list2:
        per_img_path = os.path.join(img_path, per_img_name)
        img_glasses_label = ','.join(['0','0','1','0','0','0','0','0','1','0'])

        img_glasses_name_list.append( per_img_path )
        img_glasses_label_list.append( img_glasses_label )
        fold_list.append( 0 )
'''

# sun glasses 
img_path = "./data/img/sun_glasses"
all_img_name = os.listdir(img_path)

img_sun_glasses_name_list = []
img_sun_glasses_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_sun_glasses_label = ','.join(['0','0','1','1','1','0','0','0','0','1'])

    img_sun_glasses_name_list.append( per_img_path )
    img_sun_glasses_label_list.append( img_sun_glasses_label )
    fold_list.append( 0 )
'''
img_path = "./data/img/sun_glasses_crop"
all_img_name = os.listdir(img_path)

for per_img_name in all_img_name:

    if per_img_name not in black_list2:
        per_img_path = os.path.join(img_path, per_img_name)
        img_sun_glasses_label = ','.join(['0','0','1','1','1','0','0','0','0','1'])

        img_sun_glasses_name_list.append( per_img_path )
        img_sun_glasses_label_list.append( img_sun_glasses_label )
        fold_list.append( 0 )
'''

# face mask glasses 
img_path = "./data/img/face_mask_glasses"
all_img_name = os.listdir(img_path)

img_face_mask_glasses_name_list = []
img_face_mask_glasses_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_face_mask_glasses_label = ','.join(['0','0','1','0','0','1','1','1','1','0'])

    img_face_mask_glasses_name_list.append( per_img_path )
    img_face_mask_glasses_label_list.append( img_face_mask_glasses_label )
    fold_list.append( 0 )



# face mask sun glasses 
img_path = "./data/img/face_mask_sun_glasses"
all_img_name = os.listdir(img_path)

img_face_mask_sun_glasses_name_list = []
img_face_mask_sun_glasses_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_face_mask_sun_glasses_label = ','.join(['0','0','1','1','1','1','1','1','0','1'])

    img_face_mask_sun_glasses_name_list.append( per_img_path )
    img_face_mask_sun_glasses_label_list.append( img_face_mask_sun_glasses_label )
    fold_list.append( 0 )



# face mask sun glasses 
img_path = "./data/img/occ_face_right_nose_mouth"
all_img_name = os.listdir(img_path)

img_occ_face_right_nose_mouth_name_list = []
img_occ_face_right_nose_mouth_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_occ_face_right_nose_mouth_label = ','.join(['0','0','1','0','1','1','1','0','0','0'])

    img_occ_face_right_nose_mouth_name_list.append( per_img_path )
    img_occ_face_right_nose_mouth_label_list.append( img_occ_face_right_nose_mouth_label )
    fold_list.append( 0 )

# negative img
img_path = "./data/img/a_img_negative"
all_img_name = os.listdir(img_path)

img_negative_name_list = []
img_negative_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_negative_label = ','.join(['1','0','0','0','0','0','0','0','0','0'])

    img_negative_name_list.append( per_img_path )
    img_negative_label_list.append( img_negative_label )
    fold_list.append( 0 )

# occ mouth img
img_path = "./data/img/face_occ_mouth"
all_img_name = os.listdir(img_path)

img_face_occ_mouth_list = []
img_face_occ_mouth_label_list = []
for per_img_name in all_img_name:
    per_img_path = os.path.join(img_path, per_img_name)
    img_face_occ_mouth_label = ','.join(['0','0','1','0','0','0','1','0','0','0'])

    img_face_occ_mouth_list.append( per_img_path )
    img_face_occ_mouth_label_list.append( img_face_occ_mouth_label )
    fold_list.append( 0 )

all_img_path_list = img_face_occ_mouth_list + img_negative_name_list + img_occ_hand_name_list + img_normal_name_list + img_occ_objects_name_list + img_occ_others_name_list + img_occ_hand_normal_name_list + img_face_mask_name_list + img_glasses_name_list + img_sun_glasses_name_list + img_face_mask_glasses_name_list + img_face_mask_sun_glasses_name_list + img_occ_face_right_nose_mouth_name_list
img_label_list = img_face_occ_mouth_label_list + img_negative_label_list + img_occ_hand_label_list + img_normal_label_list + img_occ_objects_label_list + img_occ_others_label_list + img_occ_hand_normal_label_list + img_face_mask_label_list + img_glasses_label_list + img_sun_glasses_label_list + img_face_mask_glasses_label_list + img_face_mask_sun_glasses_label_list + img_occ_face_right_nose_mouth_label_list


res = DataFrame()
res['filepath'] = list( all_img_path_list )
res['target'] = list( img_label_list )
res['fold'] = list( fold_list )
res[ ['filepath', 'target', 'fold'] ].to_csv('./data/data.csv', index=False)     
