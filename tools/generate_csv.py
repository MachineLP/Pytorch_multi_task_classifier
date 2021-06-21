
from pathlib import Path
import os
from pandas.core.frame import DataFrame


video_path = "./data/actions2/"
all_labels_path = os.listdir(video_path)

all_img_path_list = []
color_label_list = []
action_label_list = []
fold_list = []
for per_label_path in all_labels_path:
    if 'csv' in per_label_path:
        continue
    per_all_img_path = os.path.join(video_path, per_label_path)
    all_image_path = os.listdir(per_all_img_path)
    for per_image_path in all_image_path:
        per_img_path = os.path.join(video_path, per_label_path, per_image_path)
        all_img_path_list.append( per_img_path )
        if per_label_path.split('_')[0] == 'blue':
            color_label_list.append( 'blue' )
        if per_label_path.split('_')[0] == 'red':
            color_label_list.append( 'red' )
        if per_label_path.split('_')[0] == 'unclear':
            color_label_list.append( 'unclear' )
        if per_label_path.split('_')[0] == 'black':
            color_label_list.append( 'black' )
        
        if per_label_path.split('_')[1] == 'kick':
            action_label_list.append( 'kick' )
        if per_label_path.split('_')[1] == 'punch':
            action_label_list.append( 'punch' )
        if per_label_path.split('_')[1] == 'normal':
            action_label_list.append( 'normal' )
        
        fold_list.append( 0 )
        



res = DataFrame()
res['filepath'] = list( all_img_path_list )
res['target'] = list( color_label_list )
res['action_target'] = list( action_label_list )
res['fold'] = list( fold_list )
res[ ['filepath', 'target', 'action_target', 'fold'] ].to_csv('./data/data.csv', index=False) 





