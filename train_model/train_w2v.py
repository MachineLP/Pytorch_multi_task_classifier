# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  w2v 训练
   Author :       machinelp
   Date :         2020-06-04
-------------------------------------------------

'''
import os
import time
import jieba
import gensim
import threading
import numpy as np
from textmatch.config.config import Config as conf
from textmatch.config.constant import Constant as const
from textmatch.models.text_embedding.stop_words import StopWords


# min_count,频数阈值，大于等于1的保留
# size，神经网络 NN 层单元数，它也对应了训练算法的自由程度
# workers=4，default = 1 worker = no parallelization 只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。

if __name__ == '__main__':
    stop_word = StopWords(stopwords_file=const.STOPWORDS_FILE)
    # 训练集
    words_list = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟"]
    # doc
    words_list1 = ["我去玉龙雪山并且喜欢玉龙雪山玉龙雪山","我在玉龙雪山并且喜欢玉龙雪山","我在九寨沟", "哈哈哈哈"]
    del_stopword = True
    corpus = []
    for per_words in words_list:
        word_list = jieba.cut(per_words,cut_all=False)
        if del_stopword:
            word_list = stop_word.del_stopwords(word_list)
        corpus.append(word_list)
    
    # min_count,频数阈值，大于等于1的保留
    # size，神经网络 NN 层单元数，它也对应了训练算法的自由程度
    # workers=4，default = 1 worker = no parallelization 只有在机器已安装 Cython 情况下才会起到作用。如没有 Cython，则只能单核运行。
    model = gensim.models.Word2Vec(corpus, min_count=1, size=256)
    for per_path in [const.W2V_MODEL_FILE]:
        per_path = '/'.join(per_path.split('/')[:-1])
        if os.path.exists(per_path) == False:
            os.makedirs(per_path)
    model.save(const.W2V_MODEL_FILE)

    vector = model[corpus[-2]]
    print('vector>>>>', vector)








