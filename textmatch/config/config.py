
# 
import os
import threading

class Config():
    JIEBA_FLAG = True
    DEL_STOPWORD = False

    # 这个给定特征可以应用在 tf-idf 矩阵中，用以描述单词在文档中的最高出现率。假设一个词（term）在 80% 的文档中都出现过了，那它也许（在剧情简介的语境里）只携带非常少信息。
    MAX_DF = 0.8
    # 可以是一个整数（例如5）。意味着单词必须在 5 个以上的文档中出现才会被纳入考虑。设置为 0.2；即单词至少在 20% 的文档中出现 。
    MIN_DF = 0.2
    # 这个参数将用来观察一元模型（unigrams），二元模型（ bigrams） 和三元模型（trigrams）。参考n元模型（n-grams）。
    NGRAM_RANGE = 3 


