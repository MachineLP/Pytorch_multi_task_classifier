
TextMatch

TextMatch is a semantic matching model library for QA & text search ...  It's easy to train models and to export representation vectors.

## TODO
（1）dssm
（2）[实体识别](https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_ner_crf.py)
（3）文本纠错




# 
- wechat ID: lp9628




# 
# 

### 样例：
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python examples/text_search.py

```

examples/text_search.py
```python
import sys
from textmatch.models.text_embedding.model_factory_sklearn import ModelFactory


if __name__ == '__main__':
    # doc
    doc_dict = {"0":"我去玉龙雪山并且喜欢玉龙雪山玉龙雪山", "1":"我在玉龙雪山并且喜欢玉龙雪山", "2":"我在九寨沟", "3":"你好"}   
    # query
    query = "我在九寨沟,很喜欢"
    
    # 模型工厂，选择需要的模型加到列表中: 'bow', 'tfidf', 'ngram_tfidf', 'bert', 'albert', 'w2v'
    mf = ModelFactory( match_models=['bow', 'tfidf', 'ngram_tfidf'] )
    # 模型处理初始化
    mf.init(words_dict=doc_dict, update=True)

    # query 与 doc的相似度
    search_res = mf.predict(query)
    print ('search_res>>>>>', search_res) 
    # search_res>>>>> {'bow': [('0', 0.2773500981126146), ('1', 0.5303300858899106), ('2', 0.8660254037844388), ('3', 0.0)], 'tfidf': [('0', 0.2201159065358879), ('1', 0.46476266418455736), ('2', 0.8749225357988296), ('3', 0.0)], 'ngram_tfidf': [('0', 0.035719486884261346), ('1', 0.09654705406841395), ('2', 0.9561288696241232), ('3', 0.0)]}
    
    # query的embedding
    query_emb = mf.predict_emb(query)
    print ('query_emb>>>>>', query_emb) 
    '''
    pre_emb>>>>> {'bow': array([1., 0., 0., 1., 1., 0., 1., 0.]), 'tfidf': array([0.61422608, 0.        , 0.        , 0.4842629 , 0.4842629 ,
       0.        , 0.39205255, 0.        ]), 'ngram_tfidf': array([0.        , 0.        , 0.37156534, 0.37156534, 0.        ,
       0.        , 0.        , 0.29294639, 0.        , 0.37156534,
       0.37156534, 0.        , 0.        , 0.37156534, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.29294639, 0.37156534, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        ])}
    '''

```

### run train_model/ (train embedding(bow/tfidf/ngram tfidf/bert/albert...  train classifer))
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python train_model/train_bow.py                 (文本embedding)
python train_model/train_tfidf.py               (文本embedding)
python train_model/train_ngram_tfidf.py         (文本embedding)
python train_model/train_bert.py                (文本embedding)
python train_model/train_albert.py              (文本embedding)
python train_model/train_w2v.py                 (文本embedding)
python train_model/train_dssm.py                (文本embedding)
python train_model/train_lr_classifer.py             (文本分类)
python train_model/train_gbdt_classifer.py           (文本分类)
python train_model/train_gbdlr_classifer.py          (文本分类)
python train_model/train_lgb_classifer.py            (文本分类)
python train_model/train_xgb_classifer.py            (文本分类)
python train_model/train_dnn_classifer.py            (文本分类)
```

### run tests/core_test （qa/文本embedding）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/core_test/qa_match_test.py
python tests/core_test/text_embedding_test.py
```



### run tests/models_test （模型测试）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/models_test/bm25_test.py
python tests/models_test/edit_sim_test.py
python tests/models_test/jaccard_sim_test.py
python tests/models_test/bow_sklearn_test.py
python tests/models_test/tf_idf_sklearn_test.py
python tests/models_test/ngram_tf_idf_sklearn_test.py
python tests/models_test/w2v_test.py
python tests/models_test/albert_test.py
```

### run tests/ml_test  （机器学习模型测试）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/ml_test/lr_test.py
python tests/ml_test/gbdt_test.py
python tests/ml_test/gbdt_lr_test.py
python tests/ml_test/lgb_test.py
python tests/ml_test/xgb_test.py
```

### run tests/tools_test   （聚类/降维工具测试）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
export PYTHONPATH=${PYTHONPATH}:../TextMatch
python tests/tools_test/kmeans_test.py
python tests/tools_test/dbscan_test.py
python tests/tools_test/pca_test.py
```

### run tests/tools_test   （词云）
```
git clone https://github.com/MachineLP/TextMatch
cd TextMatch
pip install -r requirements.txt
cd tests/tools_test
python generate_word_cloud.py
```
![word_cloud](./docs/pics/word_cloud.png)
