import logging
import random
import os
import re
from pprint import pprint

import pandas as pd

from CS598_DataMining.NLPUtility import get_or_load_sentence_transformer
from CS598_DataMining.utility import set_logging

# from NaturalFundamentals.News.NLP.get_news_vector import cal_vector
pprint

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

set_logging(20)

model = get_or_load_sentence_transformer()

HOME_PATH = '/Users/yafa/Dropbox/Library/DeepLearning/CS598_DataMining/'
RAW_DATA_PATH = os.path.join(HOME_PATH, 'yelp_dataset_challenge_academic_dataset/')


def embedding(sample_rate=0.1,
              input_path=os.path.join(RAW_DATA_PATH, 'df_review'),
              output_path=os.path.join(RAW_DATA_PATH, 'df_review_subset_10.csv'),
              split_by_sentence=True):
    df_review = pd.read_csv(input_path)
    print(df_review.groupby('stars')['stars'].count())
    
    df_review_subset = df_review.loc[df_review.apply(lambda x: random.uniform(0, 1) <= sample_rate, axis=1)]
    
    df_review_subset = df_review_subset[(df_review_subset['votes_useful'] > 0) |
                                        (df_review_subset['votes_funny'] > 0) |
                                        (df_review_subset['votes_cool'] > 0)]
    
    df_review_subset = df_review_subset[df_review_subset['stars'] >= 4]
    
    if split_by_sentence:
        # Split the sentence in each review
        list_sentence = []
        for index_i, row_i in df_review_subset.iterrows():
            print(f'process: {index_i / len(df_review_subset)}')
            for sentence in re.compile('[.!\\n?;]').split(row_i['text']):
                if re.compile('[a-zA-Z]').findall(sentence):
                    if 'service' in sentence or 'Service' in sentence:
                        continue
                    list_sentence.append({
                        'sentence': sentence.strip(),
                        'review_id': row_i['review_id'],
                        'vt': model.encode(sentence.strip())
                    })
        
        df_review_by_sentence = pd.DataFrame(list_sentence)
        df_review_by_sentence = df_review_by_sentence.rename(columns={'sentence': 'text'})
        
        df_review_subset = pd.merge(df_review[['review_id', 'stars', 'business_id', 'votes_useful', 'votes_funny', 'votes_cool']],
                                    df_review_by_sentence, on=['review_id'])
        
        df_review_subset = df_review_subset.loc[df_review.apply(lambda x: random.uniform(0, 1) <= 0.5, axis=1)]
    
    
    else:
        df_review['vt'] = df_review['text'].apply(lambda x: model.encode(x))
    
    df_review_subset.to_csv(output_path)


def read_file_with_embedding(input_path=os.path.join(RAW_DATA_PATH, f'df_review_subset.csv')):
    df_review_subset = pd.read_csv(input_path)
    
    list_vt = []
    for i in df_review_subset['vt'].tolist():
        list_v = []
        for v in re.compile('[\[\]]').sub('', i).split():
            list_v.append(float(v))
        list_vt.append(list_v)
    
    df_review_subset['vt'] = list_vt
    # df_review = df_review_subset
    
    return df_review_subset

