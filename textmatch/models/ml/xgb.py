# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  XGB
   Author :       machinelp
   Date :         2020-06-13
-------------------------------------------------

'''
import sys
import logging
import numpy as np
from textmatch.config.config import cfg
from textmatch.config.constant import Constant as const
import xgboost as xgb


class XGB:

    def __init__(self):
        self.other_params = {'learning_rate': 0.125, 
                            'max_depth': 3, 
                             }
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
















