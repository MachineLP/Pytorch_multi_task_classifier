# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  DNN   trainer
   Author :       machinelp
   Date :         2020-06-06
-------------------------------------------------

'''

import json 
import numpy as np
from textmatch.models.text_embedding.model_factory_sklearn import ModelFactory
from textmatch.models.text_classifier.dnn import DNN 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping


if __name__ == '__main__':
    doc_dict = {"0":"我去玉龙雪山并且喜欢玉龙雪山玉龙雪山", "1":"我在玉龙雪山并且喜欢玉龙雪山", "2":"我在九寨沟", "3":"你好"}   #["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    #doc_dict = {"0":"This is the first document.", "1":"This is the second second document.", "2":"And the third one."}
    #query = "This is the second second document."
    query = [ "我在九寨沟,很喜欢", "我在九寨沟,很喜欢", "我在九寨沟,很喜欢"]
    
    # 基于bow
    mf = ModelFactory( match_models=['bow', 'tfidf'] )
    #mf.init(words_dict=doc_dict, update=True)
    mf.init(update=False)
    train_sample = []
    for per_query in query:
        bow_pre = mf.predict_emb(per_query)
        # print ('pre>>>>>', bow_pre)
        per_train_sample=[]
        for per_v in bow_pre.values():
           per_train_sample.extend( per_v )
        train_sample.append(per_train_sample)
    train_labels = [1,1,1]   
    #print ('train_sample, train_labels', train_sample, train_labels) 
    #print ('train_sample:::::', len(train_sample[0])) 
    train_x = np.array( train_sample[:2] )
    train_y = train_labels[:2]
    val_x = np.array(  train_sample[2:3] )
    val_y = train_labels[2:3]
    print ('train_x:', train_x)

    '''
    dnn_hidden_units = (128, 128),
    dnn_activation = 0.00001
    l2_reg_dnn = 0.00001
    dnn_dropout = 0
    dnn_use_bn = 1024
    seed = False
    dnn_input =  "text_embedding"
    inputs = Input(name='inputs',shape=[16])
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,dnn_use_bn, seed)(inputs)  
    output = tf.keras.layers.Dense( 1, use_bias=False, activation=None)(dnn_output)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    '''
    import keras as K
    init = K.initializers.glorot_uniform(seed=1)
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=128, input_dim=16, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=128, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=2, kernel_initializer=init, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",optimizer=Adam(0.001),metrics=["accuracy"])
    model_fit = model.fit(train_x,train_y,batch_size=1,epochs=10,
                      validation_data=(val_x,val_y),
                      callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)] ## 当val-loss不再提升时停止训练
                     )




