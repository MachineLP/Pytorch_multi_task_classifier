import os
import sys
import argparse
import numpy as np 
import pandas as pd 
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--data_dir', help='data path', type=str)
parser.add_argument('--n_splits', help='n_splits', type=int)
parser.add_argument('--output_dir', help='output_dir', type=str)
parser.add_argument('--random_state', help='random_state', type=int)
args = parser.parse_args()



if __name__ == '__main__':

    df_data = pd.read_csv(args.data_dir)
    img_path_list = df_data['filepath'].values.tolist()
    label1_list = df_data['target1'].values.tolist()
    label2_list = df_data['target2'].values.tolist()


    data_label = []
    video_index = []
    video_dict = {}  #...
    for m, (per_img_path, per_label1, per_label2) in enumerate(zip( img_path_list, label1_list, label2_list )):          
        data_label.append( [ per_img_path, per_label1, per_label2 ] ) 
        video_index.append( per_img_path.split("_")[-2] )
        video_dict.setdefault(per_img_path.split("_")[-2], []).append( m ) 
    video_index = list(set(video_index))

    train_list = []
    val_list = []
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state) 
    for index, (train_index, val_index) in enumerate(kf.split(video_index)): 
        for i in val_index:
            #for j, per_data_label in enumerate(data_label):
            #    # print(">>>>", per_data_label[-2].split("_")[3])
            #    if per_data_label[0].split("_")[-2] == video_index[i] :
            #        data_label[j].append(index)
            for per_index in video_dict[video_index[i]]:
                data_label[per_index].append(index)

    data_label = np.array( data_label )
    # print (data_label)


    res = DataFrame()
    res['filepath'] = data_label[:,0]
    res['target1'] = data_label[:,1]
    res['target2'] = data_label[:,2]
    res['fold'] = data_label[:,3]
    res[ ['filepath', 'target1', 'target2', 'fold'] ].to_csv(args.output_dir, index=False) 


