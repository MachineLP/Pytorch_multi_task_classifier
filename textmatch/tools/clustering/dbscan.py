# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  DBSCAN
   Author :       machinelp
   Date :         2020-06-15
-------------------------------------------------

'''

import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANClustering(object):
    def __init__(self, eps=0.5, min_samples=5):
        self.db_scan = DBSCAN(eps=eps, min_samples=min_samples)

    def predict(self, data_list):
        label_list = self.db_scan.fit_predict(data_list)
        return label_list

