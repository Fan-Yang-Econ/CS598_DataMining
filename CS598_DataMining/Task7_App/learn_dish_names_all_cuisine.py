import json
import os

import pandas as pd
from sentence_splitter import SentenceSplitter

from CS598_DataMining.Task3_DishIdentification.Task3_2_MiningAdditionalDishNames import \
    learn_dish_names, read_file_with_embedding, RAW_DATA_PATH, LIST_CUISINES
from PyHelpers import set_logging


HOME_PATH = '/Users/yafa/Dropbox/Library/DeepLearning/CS598_DataMining/CS598_DataMining/Task7_App/'

splitter = SentenceSplitter(language='en')

set_logging(20)


def get_rating_for_one_dish(df_business, df_reviews, cuisine_name, dish_name):
    # cuisine_name = 'Italian'
    df_business_ = df_business[df_business['categories'].apply(lambda x: cuisine_name in x)]
    df_review_ = df_reviews[df_reviews['business_id'].isin(df_business_['restaurant_id'])]
    
    df_review_one_dish = df_review_[df_review_['text'].apply(lambda x: dish_name.lower() in x.lower())]
    df_review_one_dish['vote_count'] = df_review_one_dish['votes_useful'] + df_review_one_dish['votes_funny'] + df_review_one_dish['votes_cool']
    df_review_one_dish['score_overall'] = df_review_one_dish['vote_count'] * df_review_one_dish['stars']
    
    return df_review_one_dish['score_overall'].mean(), len(df_review_one_dish)


def get_useful_reviews(df_business, df_reviews, cuisine_name, dish_name):
    # cuisine_name = 'Italian'
    df_business_ = df_business[df_business['categories'].apply(lambda x: cuisine_name in x)]
    df_review_ = df_reviews[df_reviews['business_id'].isin(df_business_['restaurant_id'])]
    
    df_review_one_dish = df_review_[df_review_['text'].apply(lambda x: dish_name.lower() in x.lower())]
    df_review_one_dish['vote_count'] = df_review_one_dish['votes_useful'] + df_review_one_dish['votes_funny'] + df_review_one_dish['votes_cool']
    
    df_review_one_dish = df_review_one_dish.sort_values('vote_count', ascending=False)
    
    df_good_review = df_review_one_dish[df_review_one_dish['stars'] >= 4]
    df_bad_review = df_review_one_dish[df_review_one_dish['stars'] <= 2]
    
    if not df_bad_review.empty:
        df_bad_review = df_bad_review.sort_values('votes_useful', ascending=False)
        bad_reviews = df_bad_review['text'].iloc[0]

        bad_reviews_list = splitter.split(bad_reviews)

        for position, x in enumerate(bad_reviews_list):
            if dish_name.lower() in x.lower():
                bad_reviews = ' '.join(bad_reviews_list[(position - 1): (position + 2)])
                break
    else:
        bad_reviews = None
    
    if not df_good_review.empty:
        df_good_review = df_good_review.sort_values('votes_useful', ascending=False)
        good_reviews = df_good_review['text'].iloc[0]
        
        good_reviews_list = splitter.split(good_reviews)
        
        for position, x in enumerate(good_reviews_list):
            if dish_name.lower() in x.lower():
                good_reviews = ' '.join(good_reviews_list[(position - 1): (position + 2)])
                break
    else:
        good_reviews = None
    
    return good_reviews, bad_reviews



df_reviews = read_file_with_embedding(input_path=os.path.join(RAW_DATA_PATH, f'df_review_subset_10.csv'))
#        Unnamed: 0                 user_id               review_id  stars        date                                     text             business_id  votes_useful  votes_funny  votes_cool                                       vt                                  phrases
# 21            208  AcvB6zFXu7kErZ-gafecIA  SqnQE4AULWGTyUq3GYRhog      5  2011-01-06  Our family of four dined at Vin Sant...  q8fD82us6uuGufvI44NoAg             1            0           0  [-0.10136722, 0.101143919, -0.104281...  {happy busy place, steady pace, grea...
# 22            223  vLFk6nslFe35NKvlU2Tz9Q  RsJDhYyRYuWGpJWXF_W7iQ      3  2012-12-03  Vin Santo is a good, but small Itali...  q8fD82us6uuGufvI44NoAg             3            0           0  [-0.121832639, 0.0714022294, 0.02176...  {good portions, Service, dessert, re...
# 33            336  EiOXMe4dTt94zmf5jrB0Lw  l9ywi8QLZ85vIczU0f3Vbg      3  2014-02-01  In fairness, I'd give a 3.5 for this...  ybkWtM1ZnT2ewuquj3A9KQ             0            0           0  [-0.13524802, 0.0359015316, -0.00352...  {huge pile of pasta, leftovers, wate...
# 161          1700  WRKCSRwzl6koSWHNFFc6kg  8sO_LlA8Qs5AwvCYXYUZgA      2  2006-10-19  Overcooked pasta, check. Slopped on ...  DlCtdbceo4YNSI53cCL2lg             4            8           5  [-0.100238025, 0.296311885, 0.024465...  {people, noodle, family, Average gar...
# 162          1704  ytr46wNbedr8-iqSZngl8g  CRaXlaM3wRg9n6bMRsuC0Q      1  2008-07-20  One thing that I love about OSF is t...  DlCtdbceo4YNSI53cCL2lg             2            1           2  [-0.0588928387, 0.147025481, -0.0010...  {server, thing that I hate about OSF...
#            ...                     ...                     ...    ...         ...                                      ...                     ...           ...          ...         ...                                      ...                                      ...

for vt in df_reviews['vt'].tolist():
    assert len(vt) == 768

df_business = pd.read_parquet(os.path.join(RAW_DATA_PATH, 'df_business.parquet'))

dict_dish_names = {}
for cuisine_name in LIST_CUISINES:
    if cuisine_name in dict_dish_names:
        continue
    dict_dish_names[cuisine_name] = learn_dish_names(df_review=df_reviews, df_business=df_business, cuisine_name=cuisine_name)


list_df = []
for cuisine_name in LIST_CUISINES:
    df_i = pd.DataFrame({'dish_name':dict_dish_names[cuisine_name] })
    df_i['cuisine_name'] = cuisine_name
    list_df.append(df_i)
    
df_dish = pd.concat(list_df)

# Delete dishes that appear in multiple cuisine
df_dish_score_count = df_dish.groupby(['dish_name'], as_index=False).size()
df_dish_score_count = df_dish_score_count[df_dish_score_count['size'] == 1]

df_dish = df_dish[df_dish['dish_name'].isin(df_dish_score_count['dish_name'])]


# ====== Dish Scores ======

list_dish_score = []
for cuisine_name, df_ in df_dish.groupby('cuisine_name'):
    for dish_name in df_['dish_name'].unique():
        score_, count_ = get_rating_for_one_dish(df_business, df_reviews, cuisine_name=cuisine_name, dish_name=dish_name)
        list_dish_score.append(
            {
                'dish_name': dish_name,
                'count_reviews': count_,
                'score': score_,
                'cuisine_name': cuisine_name
            }
        )

df_dish_score = pd.DataFrame(list_dish_score)
df_dish_score_sum = df_dish_score.groupby(['cuisine_name'], as_index=False)['score'].sum()
df_dish_score = pd.merge(df_dish_score, df_dish_score_sum, on=['cuisine_name'], suffixes=('', '_cuisine'))
df_dish_score['score'] = df_dish_score['score'] / df_dish_score['score_cuisine']


list_dish_score_cumsum = []
for _cuisine_name, df_dish_score_ in df_dish_score.groupby(['cuisine_name'], as_index=False):
    df_dish_score_ = df_dish_score_.sort_values('score', ascending=True)
    df_dish_score_['score'] = df_dish_score_['score'].cumsum()
    list_dish_score_cumsum.append(df_dish_score_)

df_dish_score = pd.concat(list_dish_score_cumsum)
df_dish_score = df_dish_score.sort_values('score', ascending=False)


# ====== Dish Reviews ======
list_dish_reviews = []
for cuisine_name, df_ in df_dish.groupby('cuisine_name'):
    for dish_name in df_['dish_name'].unique():
        good_reviews, bad_reviews = get_useful_reviews(df_business, df_reviews, cuisine_name=cuisine_name, dish_name=dish_name)
        print('---\n', good_reviews)
        list_dish_reviews.append(
            {
                'dish_name': dish_name,
                'bad_reviews': bad_reviews,
                'good_reviews': good_reviews,
                'cuisine_name': cuisine_name
            }
        )


df_useful_reviews = pd.DataFrame(list_dish_reviews)
df_useful_reviews = df_useful_reviews[~df_useful_reviews['bad_reviews'].isna()]
df_useful_reviews = df_useful_reviews[~df_useful_reviews['good_reviews'].isna()]

df_dish_score = pd.merge(df_dish_score, df_useful_reviews[['dish_name', 'cuisine_name']], on = ['dish_name', 'cuisine_name'])


print(json.dumps(df_useful_reviews.to_dict('records')))

with open(os.path.join(HOME_PATH, 'top_reviews.json'), '+w') as f:
    json.dump(df_useful_reviews.to_dict('records'), f)



df_dish_score['score'] = df_dish_score['score'].apply(lambda x: round(x, 5))

with open(os.path.join(HOME_PATH, 'dish_score.json'), '+w') as f:
    json.dump(df_dish_score.to_dict('records'), f)




# df_business
#                 restaurant_id  Price Range  stars                               categories
# 0      uGykseHzyS5xAMWoN6YUqA          1.0    4.0    [American (Traditional), Restaurants]
# 1      LRKJF43s9-3jG9Lgx4zODg          1.0    4.5  [Food, Ice Cream & Frozen Yogurt, Fa...
# 2      RgDg-k9S5YD_BaxMckifkg          NaN    4.0                   [Chinese, Restaurants]
