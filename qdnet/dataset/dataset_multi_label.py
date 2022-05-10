import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

import os
import cv2
import json
import pandas as pd
import numpy as np



leaf_node = ['13314', '11270', '11284', '10273', '10294', '11321', '13369', '13372', '10301', '10302', '13374', '13376', '10306', '10307', '11339', '13388', '13406', '10338', '12387', '13422', '13423', '13426', '10355', '13427', '10356', '13428', '13432', '10362', '10364', '10365', '10366', '13439', '10368', '10369', '13459', '11412', '13469', '13470', '10399', '13472', '13473', '13474', '13475', '10404', '13476', '10405', '10407', '13479', '10410', '10411', '13489', '10418', '10419', '13491', '10421', '10422', '13496', '10431', '10432', '10434', '10435', '10436', '10437', '10438', '10440', '13513', '12494', '10449', '10452', '10453', '13525', '10454', '10456', '13530', '10464', '13536', '10466', '10467', '11491', '10468', '10469', '11493', '11496', '10474', '10476', '10479', '10480', '11504', '10481', '10483', '10486', '10487', '11511', '10488', '10491', '10493', '10494', '13567', '11527', '10504', '10505', '10509', '10511', '13584', '13585', '13586', '10515', '13588', '11541', '13589', '10518', '13590', '10519', '13591', '10520', '13596', '10531', '13606', '10536', '13608', '13609', '13610', '10539', '13611', '13612', '12590', '11567', '10544', '10547', '13620', '10550', '13633', '11589', '10566', '10570', '11597', '11598', '13647', '10578', '11604', '10584', '13656', '13657', '10591', '10598', '10603', '11629', '13680', '10609', '10611', '13686', '10615', '10616', '11642', '10619', '11643', '11644', '11645', '11647', '11648', '11649', '11650', '11651', '10628', '13710', '13714', '13716', '10645', '10648', '13722', '10653', '12701', '13725', '10666', '10668', '13740', '10669', '11695', '10679', '11710', '13760', '10691', '10692', '10693', '11719', '13767', '13768', '11721', '13778', '13779', '13780', '12757', '12766', '12767', '13794', '13797', '12777', '13801', '10746', '12795', '11773', '12797', '10752', '10754', '11780', '10757', '10758', '10759', '11783', '10761', '13834', '11793', '10770', '10776', '10778', '12828', '10782', '10784', '11813', '11814', '13863', '11816', '11818', '11819', '11820', '11825', '12849', '11828', '11831', '11838', '11839', '11840', '12864', '12865', '11842', '12866', '12867', '10829', '11870', '10857', '11883', '13936', '13937', '13938', '13939', '13940', '13941', '13942', '10872', '13945', '10876', '10877', '11901', '12926', '10880', '10881', '10886', '10889', '10890', '10891', '11917', '10899', '12950', '12952', '11929', '10913', '11937', '11939', '11940', '11941', '11943', '10920', '11944', '11945', '11946', '10923', '11948', '10925', '11949', '11963', '11964', '11965', '12997', '11980', '10963', '11987', '11005', '11006', '11008', '11009', '11020', '11022', '11028', '12066', '12067', '13094', '10027', '10032', '10035', '12083', '12084', '10039', '10040', '13116', '10048', '10049', '10050', '10054', '10057', '10058', '10059', '10061', '12110', '12113', '11099', '10080', '10084', '12138', '10092', '10094', '12143', '10096', '10098', '10106', '12161', '10115', '11141', '10118', '10120', '10133', '10135', '12184', '12187', '10147', '10150', '10151', '10153', '10154', '12206', '10159', '13250', '12228', '10182', '10183', '13260', '10191', '10193', '10206', '10209', '13283', '13284', '12261', '13286', '12263', '13289', '13291', '13292', '13293', '10224', '13296', '13299', '13300', '13301', '10238']

def get_pid_list(df, id):
    tmp_df = df[df['id'] == id]

    if tmp_df.empty or tmp_df['pid'].values[0] is None:
        return pd.DataFrame(columns=['id', 'pid'])
    else:
        return get_pid_list(df, tmp_df['pid'].values[0]).append(tmp_df)



f = open("./data/category.txt")
lines = f.readlines()

id_list = []
pid_list = []
permission_source = []
for line in lines:
    per_line = line.strip().split('    ')
    id_list.append( per_line[0])
    if per_line[1] != 'null':
        pid_list.append( per_line[1] )
    else:
        pid_list.append( None )

df = pd.DataFrame(
    {
        'id': id_list,
        'pid': pid_list
    }
)

id_list = df['id'].values
father_dict = {}
for i in id_list:

    pid_list = get_pid_list(df, i)['pid'].values.tolist()
    # print(i, pid_list)
    father_dict[i] = pid_list

# 构建索引， 为每一层
label_list_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

for per_leaf in leaf_node:
    # get father list
    # 5 
    pid_list = father_dict[per_leaf] + [str(per_leaf)]
    for i in range(6):
        try:
            father_id = pid_list[i]
            if father_id not in label_list_dict[i]:
                label_list_dict[i].append( father_id )
        except:
            continue
for label_key, label_value in label_list_dict.items():
    # print ( "label_key:{}, label_value:{}".format( label_key, label_value ) )
    print ( "label_key num:{}, label_value num :{}".format( label_key, len(label_value) ) )


'''
label_key num:0, label_value num :22
label_key num:1, label_value num :146
label_key num:2, label_value num :233
label_key num:3, label_value num :77
label_key num:4, label_value num :20
label_key num:5, label_value num :11
'''
all_num = 0
for label_key, label_value in label_list_dict.items():
    all_num += len(label_value)

label_dict = {}
for label_index, per_leaf in enumerate( leaf_node ):
    # get father list
    # 5 
    pid_list = father_dict[per_leaf] + [str(per_leaf)] 
    sum = 0
    label = np.zeros([all_num,], dtype=np.float32)
    for i in range(6):
        if i < len(pid_list):
            father_id = pid_list[i]
            label[int(sum+label_list_dict[i].index(father_id))] = 1.0
            sum += len( label_list_dict[i] )
    label_dict[per_leaf] = label
    # break

print ( "label_dict>>>>>>", label_dict[str(10505)] )

class QDDataset(Dataset):
    def __init__(self, csv, mode, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor( label_dict[ str(self.csv.iloc[index].target) ] )



def get_df(data_dir):

    # train data
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_train['filepath'] = df_train['filepath']

    # test data
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_test['filepath'] = df_test['filepath']


    return df_train, df_test




