from pathlib import Path
import os
from unicodedata import category
from pandas.core.frame import DataFrame
import numpy as np


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
