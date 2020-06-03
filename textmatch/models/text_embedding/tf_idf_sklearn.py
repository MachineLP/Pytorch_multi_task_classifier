# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  tf-idf-sklearn 
   Author :       machinelp
   Date :         2020-06-03
-------------------------------------------------

'''

import os
import jieba
import pickle
import logging
import numpy as np
from .stop_words import StopWords
import textmatch.config.constant as const
from textmatch.models.model_base.model_base import ModelBase
from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class TfIdf(ModelBase):

    def __init__( self, 
                        dic_path=const.TFIDF_DIC_PATH, 
                        tfidf_model_path=const.TFIDF_MODEL_PATH, 
                        tfidf_index_path=const.TFIDF_INDEX_PATH,
                        stop_word=StopWords ):
        '''
        '''

        self.dic_path = dic_path
        self.tfidf_model_path = tfidf_model_path
        self.tfidf_index_path = tfidf_index_path
        self.stop_word = stop_word() 
        self.vectorizer = CountVectorizer(stop_words = None, token_pattern='(?u)\\b\\w\\w*\\b')    
        self.transformer = TfidfTransformer()

    # init
    def init(self, words_list=None, update=True):
        if ~os.path.exists(self.dic_path) or ~os.path.exists(self.tfidf_model_path) or ~os.path.exists(self.tfidf_index_path) or update:
            word_list = self._seg_word(words_list)
            # print ('>>>>>>>>>>', word_list)
        
        if os.path.exists(self.tfidf_model_path) and os.path.exists(self.tfidf_index_path) and update==False:
            with open(self.dic_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(self.tfidf_model_path, 'rb') as f:
                self.transformer = pickle.load(f)
        else:
            try:
                 logging.info('[Tfidf] start build tfidf model.')
                 self._gen_model(word_list)
                 logging.info('[Tfidf] build tfidf model success.')
            except Exception as e:
                 logging.error( '[Tfidf] build tfidf model error，error info: {} '.format(e) )
        
        self.words_list_pre = []
        for per_word in words_list:
            self.words_list_pre.append( self._normalize( self._predict(per_word) )[0] )
        self.words_list_pre = np.array(self.words_list_pre)
        return self
    
    # seg word
    def _seg_word(self, words_list, jieba_flag=True, del_stopword=False):
        if jieba_flag:
            word_list = [[self.stop_word.del_stopwords(words) if del_stopword else word for word in jieba.cut(words)] for words in words_list]
        else:
            word_list = [[self.stop_word.del_stopwords(words) if del_stopword else word for word in words] for words in words_list]
        return [ ' '.join(word) for word in word_list  ]

    def fit(self, word_list):
        word_list = self._seg_word(word_list)
        self._gen_model(word_list)
    
    # build dic
    def _gen_dic(self, word_list):
        dic = self.vectorizer.fit_transform(word_list)
        with open(self.dic_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        # print( 'vectorizer>>>>', self.vectorizer.get_feature_names() )
        return dic

    # build tf-idf model
    def _gen_model(self, word_list):
        tfidf = self.transformer.fit_transform(self._gen_dic(word_list))
        with open(self.tfidf_model_path, 'wb') as f:
            pickle.dump(self.transformer, f)
    
    def _predict(self, words):
        tf_idf_embedding = self.transformer.transform( self.vectorizer.transform(self._seg_word([words])) )
        # print('tf_idf_embedding0>>>>>', tf_idf_embedding)
        tf_idf_embedding = tf_idf_embedding.toarray().sum(axis=0)
        # print ('>>>>', tf_idf_embedding[np.newaxis, :]) 
        return tf_idf_embedding[np.newaxis, :].astype(float)
    
    def predict(self, words):
        pre = self._normalize( self._predict(words) )
        return np.dot( self.words_list_pre[:], pre[0] ) 

