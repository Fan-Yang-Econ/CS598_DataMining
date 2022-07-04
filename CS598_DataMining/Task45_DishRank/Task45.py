import math
import os
import logging

import pandas as pd

from CS598_DataMining.io_embedding import read_file_with_embedding, RAW_DATA_PATH, HOME_PATH
from CS598_DataMining.utility import set_logging


set_logging(20)

df_review_subset = read_file_with_embedding(input_path=os.path.join(RAW_DATA_PATH, f'df_review_subset_10.csv'))
#        Unnamed: 0                 user_id               review_id  stars        date                                     text             business_id  votes_useful  votes_funny  votes_cool                                       vt                                  phrases
# 21            208  AcvB6zFXu7kErZ-gafecIA  SqnQE4AULWGTyUq3GYRhog      5  2011-01-06  Our family of four dined at Vin Sant...  q8fD82us6uuGufvI44NoAg             1            0           0  [-0.10136722, 0.101143919, -0.104281...  {happy busy place, steady pace, grea...
# 22            223  vLFk6nslFe35NKvlU2Tz9Q  RsJDhYyRYuWGpJWXF_W7iQ      3  2012-12-03  Vin Santo is a good, but small Itali...  q8fD82us6uuGufvI44NoAg             3            0           0  [-0.121832639, 0.0714022294, 0.02176...  {good portions, Service, dessert, re...
# 33            336  EiOXMe4dTt94zmf5jrB0Lw  l9ywi8QLZ85vIczU0f3Vbg      3  2014-02-01  In fairness, I'd give a 3.5 for this...  ybkWtM1ZnT2ewuquj3A9KQ             0            0           0  [-0.13524802, 0.0359015316, -0.00352...  {huge pile of pasta, leftovers, wate...
# 161          1700  WRKCSRwzl6koSWHNFFc6kg  8sO_LlA8Qs5AwvCYXYUZgA      2  2006-10-19  Overcooked pasta, check. Slopped on ...  DlCtdbceo4YNSI53cCL2lg             4            8           5  [-0.100238025, 0.296311885, 0.024465...  {people, noodle, family, Average gar...
# 162          1704  ytr46wNbedr8-iqSZngl8g  CRaXlaM3wRg9n6bMRsuC0Q      1  2008-07-20  One thing that I love about OSF is t...  DlCtdbceo4YNSI53cCL2lg             2            1           2  [-0.0588928387, 0.147025481, -0.0010...  {server, thing that I hate about OSF...
#            ...                     ...                     ...    ...         ...                                      ...                     ...           ...          ...         ...                                      ...                                      ...

df_business = pd.read_parquet(os.path.join(RAW_DATA_PATH, 'df_business.parquet'))
# df_business
#                 restaurant_id  Price Range  stars                               categories
# 0      uGykseHzyS5xAMWoN6YUqA          1.0    4.0    [American (Traditional), Restaurants]
# 1      LRKJF43s9-3jG9Lgx4zODg          1.0    4.5  [Food, Ice Cream & Frozen Yogurt, Fa...
# 2      RgDg-k9S5YD_BaxMckifkg          NaN    4.0                   [Chinese, Restaurants]
# 3      rdAdANPNOcvUtoFgcaY9KA          2.0    3.5    [American (Traditional), Restaurants]

with open(os.path.join(HOME_PATH, 'CS598_DataMining/Task45_DishRank/student_dn_annotations.txt')) as f:
    list_dish_names = f.readlines()

list_dish_names = [i.strip() for i in list_dish_names]
print(list_dish_names)

COUSIN = 'Italian'

def generate_dish_names_info(cousin, df_business, df_review_subset, list_dish_names):

    df_business = df_business[df_business['categories'].apply(lambda x: cousin in x)]
    df_review_subset = df_review_subset[df_review_subset['business_id'].isin(df_business['restaurant_id'])]
    
    list_df = []
    for dish_name in list_dish_names:
        df_review_subset_ = df_review_subset[df_review_subset['text'].apply(lambda x: dish_name.lower() in x.lower())]
        if not df_review_subset_.empty and len(df_review_subset_) > 10:
            logging.info(f'Found dish name {dish_name}')
            df_review_subset_ = df_review_subset_[['stars', 'votes_useful', 'votes_funny', 'votes_cool', 'review_id', 'business_id']]
            df_review_subset_['dish_name'] = dish_name
            list_df.append(df_review_subset_)
    
    df_by_dish = pd.concat(list_df)
    df_by_dish['votes'] = df_by_dish['votes_useful'] + df_by_dish['votes_funny'] + df_by_dish['votes_cool']
    
    return df_by_dish


def rank_dish(df_by_dish):
    df_by_dish_sum = pd.merge(
        df_by_dish.groupby(['dish_name'], as_index=False).size(),
        df_by_dish.groupby(['dish_name'], as_index=False)['stars', 'votes_useful', 'votes_funny', 'votes_cool'].mean(),
        on=['dish_name'])
    
    df_by_dish_sum['votes'] = df_by_dish_sum['votes_useful'] + df_by_dish_sum['votes_funny'] + df_by_dish_sum['votes_cool']
    df_by_dish_sum['score'] = df_by_dish_sum['size'] * \
                              df_by_dish_sum['stars'].apply(lambda x: math.log(x)) * \
                              df_by_dish_sum['votes'].apply(lambda x: math.log(x))
    
    df_by_dish_sum = df_by_dish_sum.sort_values('score', ascending=False)
    
    df_by_dish_sum['score'] = df_by_dish_sum['score'].apply(lambda x: round(x * 100) / 100)
    
    return df_by_dish_sum.iloc[0:100][['dish_name', 'score']].to_dict('records')


rank_dish(df_by_dish)


def recommend_restaurant_by_dish(dish_name, df_by_dish):
    """
    
    :param dish_name:
        dish_name = 'margherita pizza'
    :param df_by_dish:
    :return:
    """
    
    df_by_dish_ = df_by_dish[df_by_dish['dish_name'] == dish_name]
    
    df_by_restaurant = pd.merge(
        df_by_dish_.groupby(['business_id'], as_index=False).size(),
        df_by_dish_.groupby(['business_id'], as_index=False)['stars', 'votes'].mean(),
        on=['business_id'])
    
    df_by_restaurant['score'] = df_by_restaurant['size'] * \
                                df_by_restaurant['stars'].apply(lambda x: math.log(x + 2)) * \
                                df_by_restaurant['votes'].apply(lambda x: math.log(x + 2))
    
    df_by_restaurant = df_by_restaurant.sort_values('score', ascending=False)
    
    df_by_restaurant['score'] = df_by_restaurant['score'].apply(lambda x: round(x * 100) / 100)
    df_by_restaurant.reset_index(inplace=True, drop=True)
    return df_by_restaurant


def _integration_test():
    df_by_dish = generate_dish_names_info('Italian', df_business, df_review_subset, list_dish_names=list_dish_names)
    rank_dish(df_by_dish)
    
    recommend_restaurant_by_dish('margherita pizza', df_by_dish).to_dict('records')
    recommend_restaurant_by_dish('italian sausage', df_by_dish)
