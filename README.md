
TextMatch

TextMatch is a semantic matching model library for QA & text search ...  It's easy to train models and to export representation vectors.

Let's [**Run examples**](./EXAMPLES.md) !

## Models List

|         Model       |   models   |    tests   |   pretrain  |
| :-----------------: | :---------:| :---------:|
|  Bow  | [1](./textmatch/models/text_embedding/bow_sklearn.py)                    | [test](./tests/models_test/bow_sklearn_test.py) |
| TFIDF | [2](./textmatch/models/text_embedding/tf_idf_sklearn.py)                 | [test](./tests/models_test/tf_idf_sklearn_test.py) |
| Ngram-TFIDF     | [3](./textmatch/models/text_embedding/ngram_tf_idf_sklearn.py) | [test](./tests/models_test/ngram_tf_idf_sklearn_test.py) |
| W2V     | [4](./textmatch/models/text_embedding/w2v.py)                          | [test](./tests/models_test/w2v_test.py) |
| BERT    | [5](./textmatch/models/text_embedding/bert_embedding.py)               | |
| ALBERT  | [6](./textmatch/models/text_embedding/albert_embedding.py)             |  |
| DSSM    |  |  |
| ....    |  |  |
| lr    | [92](./textmatch/models/ml/lr.py) | [test](./tests/ml_test/lr_test.py) |
| gbdt    | [93](./textmatch/models/ml/gbdt.py) | [test](./tests/ml_test/gbdt_test.py) |
| gbdt_lr    | [94](./textmatch/models/ml/gbdt_lr.py) | [test](./tests/ml_test/gbdt_lr_test.py) |
| lgb    | [95](./textmatch/models/ml/lgb.py) | [test](./tests/ml_test/lgb_test.py) |
| xgb    | [96](./textmatch/models/ml/xgb.py) | [test](./tests/ml_test/xgb_test.py) |
| Bagging    | [97](./textmatch/models/text_embedding/model_factory_sklearn.py)     | [test](./tests/models_test/factory_test.py)  |
| QA    | [98](./textmatch/core/qa_match.py)     | [test](./tests/core_test/qa_match_test.py)  |
| Text Embedding    | [99](./textmatch/core/text_embedding.py)     | [test](./tests/core_test/text_embedding_test.py)  |



## TODO
（1）dssm
（2）[实体识别](https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_ner_crf.py)




# 
- wechat ID: lp9628

