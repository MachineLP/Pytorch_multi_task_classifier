# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  KMeans
   Author :       machinelp
   Date :         2020-06-15
-------------------------------------------------

'''

import numpy as np
from sklearn.cluster import KMeans


class KMeansClustering(object):
    def __init__(self, n_clusters, random_state=9):
        self.k_means = KMeans(n_clusters=n_clusters, random_state=random_state)

    def predict(self, data_list):
        label_list = self.k_means.fit_predict(data_list)
        return label_list



