import datetime
import os

import requests

import numpy as np
import pandas as pd

from CS598_DataMining.Task6_HealthInspection.read_hygiene_data import read_data, HOME_PATH
from CS598_DataMining.io_embedding import model

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics


def onehot_encoding(df, category_var, unique_value):
    list_v = []
    for cuisine_list in df[category_var].tolist():
        if unique_value in cuisine_list:
            list_v.append(1)
        else:
            list_v.append(0)
    df[unique_value] = list_v


def submit(y_test, netid='fanyang3'):
    """
    https://www.coursera.org/learn/cs-598-dmc/supplement/zVQXW/task-6-overview
    :param y_test:
    :param netid:
    :return:
    """
    SUBMISSION_URL = 'http://capstone-leaderboard.centralus.cloudapp.azure.com'
    
    req = {
        'netid': netid,
        'alias': datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'),
        'results': [{
            'error': None,
            'dataset': 'hygiene',
            'results': list([int(i) for i in y_test])
        }]
    }
    response = requests.post(SUBMISSION_URL + '/api', json=req)
    jdata = response.json()
    
    if jdata['submission_success'] is not True:
        print("An error occurred during submission!")
        print("See the below JSON response for more information.")
        print()
        print(jdata)
    else:
        print("Submissi")


def prepare_x(df_, review_count_range, star_range, set_cuisine=None):
    df_['vt'] = df_['review'].apply(lambda x: model.encode(x))
    df_['cusine'].apply(lambda x: len(x))
    
    if set_cuisine is None:
        set_cuisine = set()
        for list_cuisine in df_['cusine'].tolist():
            for cuisine in list_cuisine:
                set_cuisine.add(cuisine)
        
        set_cuisine = set_cuisine - {'Restaurants'}
    
    for cuisine in set_cuisine:
        onehot_encoding(df=df_, category_var='cusine', unique_value=cuisine)
    
    df_['review_count_normalized'] = (df_['review_count'] - review_count_range[0]) / (review_count_range[1] - review_count_range[0])
    df_['avg_star_normalized'] = (df_['avg_star'] - star_range[0]) / (star_range[1] - star_range[0])
    
    list_x = []
    for _, i in df_.iterrows():
        x = i['vt'].tolist()
        for cuisine in set_cuisine:
            x.append(i[cuisine])
        x.append(i['review_count_normalized'])
        x.append(i['avg_star_normalized'])
        
        list_x.append(np.array(x))

    X = np.array(list_x)
    return X, set_cuisine


df_train, df_test = read_data(data_folder=os.path.join(HOME_PATH, 'yelp_dataset_challenge_academic_dataset/YelpHygieneData/'))

review_count_range = df_train['review_count'].min(), df_train['review_count'].max()
star_range = df_train['avg_star'].min(), df_train['avg_star'].max()

X, set_cuisine = prepare_x(df_=df_train, review_count_range=review_count_range, star_range=star_range)

len(set_cuisine)

X_TEST, set_cuisine_test = prepare_x(df_=df_test, review_count_range=review_count_range, star_range=star_range, set_cuisine=set_cuisine)

print(len(X_TEST[0]))

clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=1000)
clf.fit(X, df_train['label'])

print(metrics.f1_score(df_train['label'], clf.predict(X), pos_label=1))

clf_gdb = GradientBoostingClassifier(max_depth=3, random_state=0, n_estimators=50, learning_rate=0.1)
clf_gdb.fit(X, df_train['label'])
print(metrics.f1_score(df_train['label'], clf_gdb.predict(X), pos_label=1))


y_test = clf.predict(X_TEST)
y_test.mean()

pd.DataFrame({'y_pred': y_test}).to_csv(os.path.join(HOME_PATH, 'CS598_DataMining/Task6_HealthInspection/y_prediction_test.csv'), index=False)

y_test = pd.read_csv(os.path.join(HOME_PATH, 'CS598_DataMining/Task6_HealthInspection/y_prediction_test.csv'))['y_pred'].tolist()

submit(y_test, netid='fanyang3')
