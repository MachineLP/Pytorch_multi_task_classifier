from pathlib import Path
import os
from unicodedata import category
from pandas.core.frame import DataFrame
import numpy as np
balck_list = ['168.jpg','1188.jpg','1189.jpg','1650.jpg','1664.jpg','2003.jpg','3099.jpg','4081.jpg','4239.jpg','6425.jpg','6513.jpg','6536.jpg','7439.jpg','7804.jpg','7856.jpg','7905.jpg','8265.jpg','9388.jpg','9418.jpg','9793.jpg','9927.jpg','10418.jpg','10433.jpg','10770.jpg','11805.jpg','13869.jpg','13872.jpg','13873.jpg','13910.jpg','14239.jpg','14245.jpg','14321.jpg','14534.jpg','16403.jpg','16416.jpg','16438.jpg','17058.jpg','17683.jpg','18757.jpg','20679.jpg','20737.jpg','22313.jpg','22325.jpg','24755.jpg','24790.jpg','24814.jpg','24825.jpg','25080.jpg','25268.jpg','25921.jpg','26556.jpg','26565.jpg','26569.jpg','26582.jpg','27039.jpg','27462.jpg','27477.jpg','27538.jpg','27830.jpg','27909.jpg','28377.jpg','28438.jpg','28643.jpg','28661.jpg','28692.jpg','28693.jpg','28698.jpg','28738.jpg','28987.jpg','29344.jpg','29508.jpg','29652.jpg','29858.jpg','29856.jpg']

f = open("./data/data_label.txt")
lines = f.readlines()
img_occ_name_list = []
img_normal_name_list = []
img_occ_label_list = []
img_normal_label_list = []
fold_list = []
for line in lines:
    per_line = line.strip().split(',')
    img_name = per_line[0]
    if img_name in balck_list:
        continue
    occ_label_list= ','.join( per_line[1:] )
    # occ_label_list = [ int(i) for i in per_line[1:] ]
    img_occ_label = np.array( occ_label_list + ['0','0','0'] )
    img_normal_label = np.array( ['0','0','0','0','0','0','0','0'] )

    img_occ_name_list.append( './data/occ_path/' + img_name )
    img_normal_name_list.append( './data/normal_path/' + img_name )
    img_occ_label_list.append( img_occ_label )
    img_normal_label_list.append( img_normal_label )
    fold_list.append( 0 )


all_img_path_list = img_occ_name_list + img_normal_name_list
img_label_list = img_occ_label_list + img_normal_label_list
fold_list = fold_list + fold_list

res = DataFrame()
res['filepath'] = list( all_img_path_list )
res['target'] = list( img_label_list )
res['fold'] = list( fold_list )
res[ ['filepath', 'target', 'fold'] ].to_csv('./data/data.csv', index=False)     
