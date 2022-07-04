"""
# Task 2.1: Visualization of the Cuisine Map
#
# Use all the reviews of restaurants of each cuisine to represent that cuisine and compute the similarity of cuisines based on the similarity
# of their corresponding text representations. Visualize the similarities of the cuisines and describe your visualization.
#
# The visualization shows the similarity matrix, with every cell corresponding to the similarity between two cuisines.
# The opacity of each cell is the similarity - with a higher opacity for a higher similarity.


# Task 2.2: Improving Cuisine Maps
#
# Try to improve the cuisine map by 1) varying the text representation
# (e.g., improving the weighting of terms or applying topic models) and
# 2) varying the similarity function
# (e.g., concatenate all reviews then compute the similarity vs. computing similarity of individual reviews and then aggregate the similarity values).
# Does any improvement lead to a better map?


# Task 2.3: Incorporating Clustering Cuisine Maps
#
# Use any similarity results from Task 2.1 or Task 2.2 to do clustering.
#
# Visualize the clustering results to show the major categories of cuisines.
#
# Vary the number of clusters to try at least two very different numbers of clusters,
# and discuss how this affects the quality or usefulness of the map. Use multiple clustering algorithms for this task.
#
# Also note in that each color is a separate cluster in the sample images below.


# Clustering method: IDF, NonIDF, LDA

"""

import os
from pprint import pprint
import pandas as pd
import numpy as np

import sklearn.cluster

from CS598_DataMining.utility import cal_cosine_simi
from CS598_DataMining.Task1_LearnTopics.learn_topics import add_noun_bunks_to_df
from CS598_DataMining.io_embedding import read_file_with_embedding, RAW_DATA_PATH
from CS598_DataMining.Task2_CusineClustering.processYelpRestaurants import OS_PATH as OS_PATH_RAW_DATA
from CS598_DataMining.utility import set_logging

set_logging(20)

df_review_subset = read_file_with_embedding(input_path=os.path.join(RAW_DATA_PATH, f'df_review_subset_10.csv'))
for vt in df_review_subset['vt'].tolist():
    assert len(vt) == 768

df_review_count = df_review_subset.groupby('business_id', as_index=False)['review_id'].nunique()

df_business = pd.read_parquet(os.path.join(OS_PATH_RAW_DATA, 'df_business.parquet'))

list_row = []
for _, row_i in df_business.iterrows():
    for cate in row_i['categories']:
        row_i['category'] = cate
        list_row.append(row_i)

df_business_per_cate = pd.DataFrame(list_row)
df_business_per_cate = df_business_per_cate[df_business_per_cate['category'] != 'Restaurants']
pprint(df_business_per_cate['category'].unique())
series_count = df_business_per_cate.groupby('category', as_index=True)['category'].count()

list_valid_categories = list(series_count[series_count > 10].index)
list_valid_categories = [i for i in list_valid_categories if i not in ['Caterers', 'Event Planning & Services', 'Hotels']]

print(f'list_valid_categories {list_valid_categories}')

df_business['categories'] = df_business['categories'].apply(lambda x: set(x).intersection(set(list_valid_categories)))
df_business = df_business[df_business['categories'].apply(lambda x: len(x)) > 0]

# Add categories to reviews
df_review_subset = pd.merge(df_review_subset, df_business[['categories', 'restaurant_id']].rename(columns={'restaurant_id': 'business_id'}),
                            on='business_id')

df_review_subset = add_noun_bunks_to_df(df_review_subset)


def learn_cuisine_embedding_simple(list_valid_categories, df_review_subset):
    """
    
    :param list_valid_categories:
        ['African', 'Brazilian', 'Burgers', 'Cafes', 'Fast Food', 'Gastropubs', 'Italian', 'Nightlife', 'Pizza', 'Sandwiches', 'Specialty Food', 'Tex-Mex', 'Turkish']
 
    :param df_review_subset:
        df_review_subset
       Unnamed: 0                 user_id               review_id  stars        date                                     text             business_id  votes_useful  votes_funny  votes_cool                                       vt                        categories
0              39  _xf2ECTRftPV4bzK4mGYEg  LFmGd7MWHPdXxx0MqNc1Eg      3  2009-09-03  Great little bar, friendly bartender...  zOc8lbjViUZajbY7M0aUCQ             1            0           0  [0.022307016, 0.0491050258, 0.017433...                           {Pizza}
1              50  XxWsOrQp0gCzDmvOsGvYyA  Yi2O65hivy_kzn8iSJnleg      2  2010-07-05  Took four kids tonight and we were a...  SKLw05kEIlZcpTD5pqma8Q             2            1           1  [0.0277398061, 0.203517497, -0.08034...       {Event Planning & Services}
2              52  Orf6p06pNjMHuPXypLbn-w  9mnSxTZlMmtt7uWj_nLsJQ      5  2011-10-30  This place is wonderful.  One, Almos...  SKLw05kEIlZcpTD5pqma8Q             0            0           0  [-0.00952077843, 0.020613391, -0.008...       {Event Planning & Services}
 
    :return:
    """
    cate_embedding = {}
    for cate in list_valid_categories:
        
        df_review_subset_ = df_review_subset[df_review_subset['categories'].apply(lambda x: cate in x)]
        if df_review_subset_.empty:
            continue
        
        vt_sum = None
        for vt in df_review_subset_['vt'].tolist():
            # len(vt)
            # len(vt_sum)
            if not vt_sum:
                vt_sum = vt
            else:
                print(len(vt_sum))
                vt_sum = [i + vt[index_i] for index_i, i in enumerate(vt_sum)]
        
        vt_avg = [int(i / len(df_review_subset_) * 1000) / 1000 for i in vt_sum]
        
        cate_embedding[cate] = {'embedding': vt_avg}
    
    return cate_embedding


cate_embedding = learn_cuisine_embedding_simple(list_valid_categories, df_review_subset)


def generate_heat_map_data_source(cate_embedding):
    list_sim = []
    for cate_i in cate_embedding:
        for cate_j in cate_embedding:
            list_sim.append(
                {
                    'x': cate_i,
                    'y': cate_j,
                    'v': cal_cosine_simi(cate_embedding[cate_i]['embedding'], cate_embedding[cate_j]['embedding'])
                }
            )
    
    df_sim = pd.DataFrame(list_sim)
    
    print({
        'min_value': df_sim['v'].min() * 0.8,
        'max_value': 1
    })
    
    return df_sim


def learn_cuisine_embedding_simple(list_valid_categories, df_review_subset):
    """

    :param list_valid_categories:
        ['African', 'Brazilian', 'Burgers', 'Cafes', 'Fast Food', 'Gastropubs', 'Italian', 'Nightlife', 'Pizza', 'Sandwiches', 'Specialty Food', 'Tex-Mex', 'Turkish']

    :param df_review_subset:
        df_review_subset
       Unnamed: 0                 user_id               review_id  stars        date                                     text             business_id  votes_useful  votes_funny  votes_cool                                       vt                        categories
0              39  _xf2ECTRftPV4bzK4mGYEg  LFmGd7MWHPdXxx0MqNc1Eg      3  2009-09-03  Great little bar, friendly bartender...  zOc8lbjViUZajbY7M0aUCQ             1            0           0  [0.022307016, 0.0491050258, 0.017433...                           {Pizza}
1              50  XxWsOrQp0gCzDmvOsGvYyA  Yi2O65hivy_kzn8iSJnleg      2  2010-07-05  Took four kids tonight and we were a...  SKLw05kEIlZcpTD5pqma8Q             2            1           1  [0.0277398061, 0.203517497, -0.08034...       {Event Planning & Services}
2              52  Orf6p06pNjMHuPXypLbn-w  9mnSxTZlMmtt7uWj_nLsJQ      5  2011-10-30  This place is wonderful.  One, Almos...  SKLw05kEIlZcpTD5pqma8Q             0            0           0  [-0.00952077843, 0.020613391, -0.008...       {Event Planning & Services}

    :return:
    """
    
    cate_embedding_simple = {}
    for cate in list_valid_categories:
        print(f'Category `{cate}`')
        df_review_subset_ = df_review_subset[df_review_subset['categories'].apply(lambda x: cate in x)]
        if df_review_subset_.empty:
            continue
        
        weight = 0
        vt_sum = None
        for _, row_i in df_review_subset_.iterrows():
            # len(vt)
            # len(vt_sum)
            vt = pd.Series(row_i['vt'])
            
            weight_i = (1 + (row_i['votes_useful'] + row_i['votes_funny'] + row_i['votes_cool']) * 0.1)
            weight += weight_i
            
            vt = weight_i * vt
            if not vt_sum:
                vt_sum = vt.tolist()
            else:
                vt_sum = [i + vt[index_i] for index_i, i in enumerate(vt_sum)]
        
        vt_avg_simple = [int(i / weight * 1000) / 1000 for i in vt_sum]
        
        cate_embedding_simple[cate] = {'embedding': vt_avg_simple}
    
    cate_embedding = {}
    for cate in list_valid_categories:
        print(f'Category `{cate}`')
        df_review_subset_ = df_review_subset[df_review_subset['categories'].apply(lambda x: cate in x)]
        
        if df_review_subset_.empty:
            continue
        
        vt_sum = None
        weight = 0
        for index_, row_i in df_review_subset_.iterrows():
            # len(vt)
            # len(vt_sum)
            
            vt = row_i['vt']
            
            list_sim_with_other_topics = []
            for cate_other in cate_embedding_simple:
                if cate_other == cate:
                    continue
                else:
                    list_sim_with_other_topics.append(cal_cosine_simi(vt, cate_embedding_simple[cate_other]['embedding']))
            
            commonality = pd.Series(list_sim_with_other_topics).mean()
            weight_i = (1 / commonality) ** 2
            weight += weight_i
            
            vt = pd.Series(vt)
            vt = weight_i * vt
            
            if not vt_sum:
                vt_sum = vt.tolist()
            else:
                vt_sum = [i + vt[index_i] for index_i, i in enumerate(vt_sum)]
        
        vt_avg_simple = [int(i / weight * 1000) / 1000 for i in vt_sum]
        
        cate_embedding[cate] = {'embedding': vt_avg_simple}
    
    return cate_embedding


df_cate_sim = generate_heat_map_data_source(cate_embedding)
df_cate_embedding = pd.DataFrame(cate_embedding).transpose()
df_cate_embedding['cate'] = df_cate_embedding.index

df_cate_sim['cluster_score']

k_means = sklearn.cluster.KMeans(n_clusters=3,
                                 init='k-means++',
                                 n_init=10, max_iter=300,
                                 tol=0.0001, verbose=0,
                                 random_state=1,
                                 copy_x=True, algorithm='auto')

X = df_cate_embedding['embedding'].tolist()
X = np.array(X)
for index_v, v in enumerate(X):
    X[index_v] = np.array(v)

k_means.fit(X)

len(X)
len(k_means.fit_predict(X))

df_cate_embedding['labels'] = k_means.fit_predict(X)
df_cate_embedding = df_cate_embedding.sort_values('labels')

df_cate_sim['labels'] = df_cate_sim['x'].apply(lambda x: df_cate_embedding[df_cate_embedding['cate'] == x]['labels'].iloc[0])
df_cate_sim
print(df_review_subset.groupby(['labels'])['labels'].count())

df_cate_sim = df_cate_sim[df_cate_sim['v'] != 1]
df_cate_sim['v'] > 0.9

print(cate_embedding)

from sklearn.decomposition import PCA  # Principal Component Analysis

pca_2d = PCA(n_components=3)
df_2d = pd.DataFrame(pca_2d.fit_transform(X))

df_2d
X = df_cate_embedding['embedding'].tolist()
X = np.array(X)
for index_v, v in enumerate(X):
    X[index_v] = np.array(v)

k_means.fit(df_2d)

df_2d.columns = ['x', 'y']

df_cate_embedding = df_cate_embedding.reset_index()
df_2d = df_2d.reset_index()

df_cate_embedding = pd.concat([df_cate_embedding, df_2d], axis=1)
df_cate_embedding['labels'] = df_cate_embedding['labels'].apply(lambda x: 'Cluster' + str(x))
from plotnine import *

ggplot(df_cate_embedding) + geom_point(
    aes(x='x', y='y', color='labels'))
