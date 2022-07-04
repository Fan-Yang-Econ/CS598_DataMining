import re

import sklearn.cluster
import numpy as np
import pandas as pd

from CS598_DataMining.utility import cal_cosine_simi
from CS598_DataMining.NLPUtility import get_or_load_spacy_model, get_cleaned_noun_chunks, get_or_load_sentence_transformer


def learn_topics(df_review_subset, n_topics=4):
    X = df_review_subset['vt'].tolist()
    X = np.array(X)
    for index_v, v in enumerate(X):
        X[index_v] = np.array(v)
    
    k_means = sklearn.cluster.KMeans(n_clusters=n_topics,
                                     init='k-means++',
                                     n_init=10, max_iter=300,
                                     tol=0.0001, verbose=0,
                                     random_state=1,
                                     copy_x=True, algorithm='auto')
    
    k_means.fit(X)
    df_review_subset['labels'] = k_means.fit_predict(X)
    print(df_review_subset.groupby(['labels'])['labels'].count())

    return df_review_subset
    

def add_noun_bunks_to_df(df_review_subset):
    
    list_noun_chunk = []
    for index_h, h in enumerate(df_review_subset['text'].tolist()):
        
        print(index_h / len(h))
        
        x = get_or_load_spacy_model(model_name="en_core_web_md")(h)
        
        _list_noun_chunk_info = get_cleaned_noun_chunks(
            x,
            remove_special_entity_types=('DATE', 'TIME', 'MONEY', 'QUANTITY', 'PERCENT', 'CARDINAL'),
            remove_stop_word_start=True,
            simple_output=False)
        _list_noun_chunk = [i['_TEXT_'] for i in _list_noun_chunk_info]
        list_noun_chunk.append(_list_noun_chunk)
    
    df_review_subset['noun_chunks'] = list_noun_chunk
    return df_review_subset
    

def learn_top_noun_chunks_by_topic(df_review_subset):
    
    DICT_NOUN_CHUNK_BY_TOPIC = {}
    
    for topic, df_review_one_topic in df_review_subset.groupby('labels'):
        
        vt = None
        for vt_ in df_review_one_topic['vt'].tolist():
            if vt is None:
                vt = pd.Series(vt_)
            else:
                vt = vt + pd.Series(vt_)
        
        topic_vt = (vt / len(df_review_one_topic)).tolist()
        df_review_one_topic['sim'] = df_review_one_topic['vt'].apply(lambda x: cal_cosine_simi(x, topic_vt))
        df_review_one_topic = df_review_one_topic.sort_values('sim', ascending=False)
        df_review_one_topic_top100 = df_review_one_topic.iloc[0:100, ]
        
        list_noun_chunk_one_topic = []
        for list_noun_chunk_ in df_review_one_topic_top100['noun_chunks'].tolist():
            list_noun_chunk_one_topic.extend(list_noun_chunk_)
        
        df_noun_chunk = pd.DataFrame({'noun_chunks': list_noun_chunk_one_topic}).drop_duplicates('noun_chunks')
        df_noun_chunk['noun_chunks'] = df_noun_chunk['noun_chunks'].apply(lambda x: re.compile('’').sub("'", x))
        
        df_noun_chunk = df_noun_chunk[df_noun_chunk['noun_chunks'].apply(lambda x: len(x.split()) > 1)]
        
        df_noun_chunk['vt'] = df_noun_chunk['noun_chunks'].apply(lambda x: get_or_load_sentence_transformer().encode(x, show_progress_bar=False))
        df_noun_chunk['sim'] = df_noun_chunk['vt'].apply(lambda x: cal_cosine_simi(x, topic_vt))
        df_noun_chunk = df_noun_chunk.sort_values('sim', ascending=False)
        
        print(df_noun_chunk.iloc[0:100, ]['noun_chunks'].tolist())
        
        df_noun_chunk_top = df_noun_chunk.iloc[0:5, ]
        
        DICT_NOUN_CHUNK_BY_TOPIC[topic] = df_noun_chunk_top
    
    return DICT_NOUN_CHUNK_BY_TOPIC

