import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

from tqdm import tqdm


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
            return data, {'color': torch.tensor(self.csv.iloc[index].target1).long(), 'action': torch.tensor(self.csv.iloc[index].target2).long()}



def get_df(data_dir):

    # train data
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_train['filepath'] = df_train['filepath']

    # test data
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_test['filepath'] = df_test['filepath']

    # class mapping
    label2idx = {d: idx for idx, d in enumerate(sorted(df_train.target1.unique()))}
    print ("target1 label2idx:", label2idx)
    df_train['target1'] = df_train['target1'].map(label2idx)

    label2idx = {d: idx for idx, d in enumerate(sorted(df_train.target2.unique()))}
    print ("target2 label2idx:", label2idx)
    df_train['target2'] = df_train['target2'].map(label2idx)

    return df_train, df_test

