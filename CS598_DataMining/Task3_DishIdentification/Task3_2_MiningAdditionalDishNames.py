"""
Learn new dishes from the Italian Cuisine
"""

import os
import pickle
import re
import typing
import pandas as pd

from CS598_DataMining.NLPUtility.CONFIG import get_or_load_sentence_transformer
from CS598_DataMining.NLPUtility.entity import get_cleaned_noun_chunks, get_or_load_spacy_model
from CS598_DataMining.io_embedding import read_file_with_embedding, RAW_DATA_PATH
from CS598_DataMining.utility import set_logging

HOME_PATH = '/Users/yafa/Dropbox/Library/DeepLearning/CS598_DataMining/CS598_DataMining/Task3_DishIdentification/'

LIST_CUISINES = ['American (New)',
                 'Mediterranean',
                 'Chinese',
                 'Indian',
                 'Italian']


def find_unique_phrases(text):
    list_text = get_cleaned_noun_chunks(
        get_or_load_spacy_model()(text),
        remove_special_entity_types=('DATE', 'TIME', 'MONEY', 'QUANTITY', 'PERCENT', 'CARDINAL',
                                     'ORDINAL', 'ORG', 'LOC', 'FAC', 'PERSON', 'NORP'),
        simple_output=True
    )
    
    set_text = set(list_text)
    
    set_phrase_filtered = set()
    for i in set_text:
        if i not in '---'.join(set_text - {i}):
            set_phrase_filtered.add(i)
    
    return set_phrase_filtered


def get_valid_phrases(df_review_subset, min_support=6):
    df_review_subset['phrases'] = df_review_subset['text'].apply(lambda x: find_unique_phrases(x))
    
    list_phrases = []
    for set_phrases in df_review_subset['phrases'].tolist():
        list_phrases.extend(list(set_phrases))
    
    df_phrases = pd.DataFrame({'phrase': list_phrases})
    df_phrases['phrase_lower'] = df_phrases['phrase'].apply(lambda x: x.lower())
    
    df_phrases_size = df_phrases.groupby(['phrase_lower'], as_index=False).size()
    df_phrases_size = df_phrases_size[df_phrases_size['size'] >= min_support]
    
    set_text = set(df_phrases_size['phrase_lower'].unique())
    
    set_phrase_filtered = set()
    for i in set_text:
        if i not in '---'.join(set_text - {i}):
            set_phrase_filtered.add(i)
    
    return set_phrase_filtered


def learn_dish_names(df_review, df_business, cuisine_name='Italian') -> typing.List[str]:
    """
    
    :param df_review_subset:
    :param df_business:
    :param cuisine_name:
    :return:
    """
    
    df_business_ = df_business[df_business['categories'].apply(lambda x: cuisine_name in x)]
    df_review_ = df_review[df_review['business_id'].isin(df_business_['restaurant_id'])]
    
    set_phrase_filtered = get_valid_phrases(df_review_, min_support=6)
    
    len(set_phrase_filtered)
    
    df_phrase = pd.DataFrame({'phrase': list(set_phrase_filtered)})
    # df_phrase = df_phrase.iloc[0:1000]
    df_phrase['vt'] = df_phrase['phrase'].apply(lambda x: get_or_load_sentence_transformer().encode(x))
    
    df_feature_new_phrase = pd.DataFrame(df_phrase['vt'].tolist())
    
    # Add the additional feature:
    for additional_col in LIST_CUISINES:
        additional_col = 'cuisine_' + additional_col
        additional_col = re.compile(' +').sub('_', additional_col)
        df_feature_new_phrase[additional_col] = 0
    
    target_cuisine = 'cuisine_' + cuisine_name
    target_cuisine = re.compile(' +').sub('_', target_cuisine)
    df_feature_new_phrase[target_cuisine] = 1
    # df_feature_new_phrase.keys()
    
    with open(os.path.join(HOME_PATH, 'random_forest_model.pickle'), 'rb') as f:
        random_forest_model = pickle.load(f)
    
    predicted_dish_boolean = random_forest_model.predict(df_feature_new_phrase)
    
    # Phrases newly learned
    
    return df_phrase.iloc[list(predicted_dish_boolean == 1)]['phrase'].unique().tolist()


def main():
    set_logging(20)
    
    df_review = read_file_with_embedding(input_path=os.path.join(RAW_DATA_PATH, f'df_review_subset_10.csv'))
    #        Unnamed: 0                 user_id               review_id  stars        date                                     text             business_id  votes_useful  votes_funny  votes_cool                                       vt                                  phrases
    # 21            208  AcvB6zFXu7kErZ-gafecIA  SqnQE4AULWGTyUq3GYRhog      5  2011-01-06  Our family of four dined at Vin Sant...  q8fD82us6uuGufvI44NoAg             1            0           0  [-0.10136722, 0.101143919, -0.104281...  {happy busy place, steady pace, grea...
    # 22            223  vLFk6nslFe35NKvlU2Tz9Q  RsJDhYyRYuWGpJWXF_W7iQ      3  2012-12-03  Vin Santo is a good, but small Itali...  q8fD82us6uuGufvI44NoAg             3            0           0  [-0.121832639, 0.0714022294, 0.02176...  {good portions, Service, dessert, re...
    # 33            336  EiOXMe4dTt94zmf5jrB0Lw  l9ywi8QLZ85vIczU0f3Vbg      3  2014-02-01  In fairness, I'd give a 3.5 for this...  ybkWtM1ZnT2ewuquj3A9KQ             0            0           0  [-0.13524802, 0.0359015316, -0.00352...  {huge pile of pasta, leftovers, wate...
    # 161          1700  WRKCSRwzl6koSWHNFFc6kg  8sO_LlA8Qs5AwvCYXYUZgA      2  2006-10-19  Overcooked pasta, check. Slopped on ...  DlCtdbceo4YNSI53cCL2lg             4            8           5  [-0.100238025, 0.296311885, 0.024465...  {people, noodle, family, Average gar...
    # 162          1704  ytr46wNbedr8-iqSZngl8g  CRaXlaM3wRg9n6bMRsuC0Q      1  2008-07-20  One thing that I love about OSF is t...  DlCtdbceo4YNSI53cCL2lg             2            1           2  [-0.0588928387, 0.147025481, -0.0010...  {server, thing that I hate about OSF...
    #            ...                     ...                     ...    ...         ...                                      ...                     ...           ...          ...         ...                                      ...                                      ...
    
    for vt in df_review['vt'].tolist():
        assert len(vt) == 768
    
    df_business = pd.read_parquet(os.path.join(RAW_DATA_PATH, 'df_business.parquet'))
    # df_business
    #                 restaurant_id  Price Range  stars                               categories
    # 0      uGykseHzyS5xAMWoN6YUqA          1.0    4.0    [American (Traditional), Restaurants]
    # 1      LRKJF43s9-3jG9Lgx4zODg          1.0    4.5  [Food, Ice Cream & Frozen Yogurt, Fa...
    # 2      RgDg-k9S5YD_BaxMckifkg          NaN    4.0                   [Chinese, Restaurants]
    
    # cuisine_name = 'American (New)'
    print(learn_dish_names(df_review, df_business, cuisine_name='Italian'))
