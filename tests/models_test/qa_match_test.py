# -*- coding:utf-8 -*-
'''
-------------------------------------------------
   Description :  QA测试
   Author :       liupeng
   Date :         2019-03-31
-------------------------------------------------

'''

import sys
from textmatch.core.qa_match import QMatch, AMatch, SemanticMatch

test_dict = {"id0": "购买的商品已经拒签原路退回，什么时候给我办理退换货？",
   "id1": "支付宝怎么搜索不到来分期了？/支付宝怎么不能还款了？",
   "id2": "你们来分期和支付宝是什么关系？",
   "id3": "购买的商品是假货已经开具了检测报告",
   "id4": "来分期怎么下不了单/无法借款",
   "id5": "我需要提前结清"}

def test_q_match(testword):
    # QMatch
    q_match = QMatch( q_dict=test_dict, match_models=['bow', 'tfidf', 'ngram_tfidf']) 
    q_match_pre = q_match.predict(testword, match_strategy='score', vote_threshold=0.5, key_weight = {'bow': 1, 'tfidf': 1, 'ngram_tfidf': 1})
    print ('q_match_pre>>>>>', q_match_pre )
    return q_match_pre

def test_a_match(testword):
    # AMatch
    a_match = AMatch( a_dict=test_dict, match_models=['bow', 'tfidf', 'ngram_tfidf']) 
    a_match_pre = a_match.predict(testword, ['id0', 'id1'], match_strategy='score', vote_threshold=0.5, key_weight = {'bow': 1, 'tfidf': 1, 'ngram_tfidf': 1}) 
    print ('a_match_pre>>>>>', a_match_pre )
    # a_match_pre>>>>> {'id0': 1.0, 'id1': 0.0} 
    return a_match_pre


def test_semantic_match(testword,words_dict=test_dict):
    # SemanticMatch
    s_match = SemanticMatch( words_dict=words_dict, match_models=['bow', 'tfidf', 'ngram_tfidf'] ) 
    s_match_pre = s_match.predict(testword, ['id0','id1', "id5"], match_strategy='score', vote_threshold=0.5, key_weight = {'bow': 1, 'tfidf': 1, 'ngram_tfidf': 1})
    print ('s_match_pre>>>>>', s_match_pre ) 
    # s_match_pre>>>>> {'id0': 1.0, 'id1': 0.0}
    return s_match_pre




if __name__ == '__main__':
    testword = "购买的商品已经拒签原路退回"
    test_q_match(testword)
    test_a_match(testword)
    test_semantic_match(testword)
    





