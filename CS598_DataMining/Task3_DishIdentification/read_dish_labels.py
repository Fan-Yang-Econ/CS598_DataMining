import os
import re

import numpy as np
import pandas as pd
from os import listdir
from sklearn.ensemble import RandomForestClassifier
from os.path import isfile, join

from CS598_DataMining.NLPUtility.CONFIG import get_or_load_sentence_transformer


def onehot_encoding(df, category_var):
    list_v = []
    for v in df[category_var].tolist():
        list_v.append({
            f'{category_var}_{v}': 1
        })
    
    df_onehot_encoding = pd.DataFrame(list_v)
    del df_onehot_encoding[df_onehot_encoding.columns[-1]]
    for i in df_onehot_encoding:
        df_onehot_encoding.loc[df_onehot_encoding[i].isna(), i] = 0
    return df_onehot_encoding


HOME_PATH = '/CS598_DataMining/Task3_DishIdentification/'

list_files = listdir(os.path.join(HOME_PATH, 'manualAnnotationTask'))

list_df_raw_label = []

for file in list_files:
    with open(os.path.join(HOME_PATH, f'manualAnnotationTask/{file}')) as f:
        _raw_labels = f.readlines()
    
    _raw_labels = [re.compile('[\n]').sub('', i).split('\t') for i in _raw_labels]
    _df_raw_label = pd.DataFrame(
        {'dish': [i[0] for i in _raw_labels],
         'label': [int(i[1]) for i in _raw_labels]
         }
    )
    _df_raw_label['cuisine'] = re.compile(r'\.label').sub('', file)
    
    list_df_raw_label.append(_df_raw_label)

df_raw_label = pd.concat(list_df_raw_label)

model = get_or_load_sentence_transformer()

list_embedding = []
for dish in df_raw_label['dish'].tolist():
    list_embedding.append(np.array([round(i, 3) for i in model.encode(dish)]))

df_raw_label['vt'] = list_embedding
df_cuisine_hot_encoding = onehot_encoding(df_raw_label, 'cuisine')

df_raw_label = df_raw_label.reset_index(drop=True)
df_cuisine_hot_encoding = df_cuisine_hot_encoding.reset_index(drop=True)

df_features = pd.DataFrame(df_raw_label['vt'].tolist())
df_features = pd.concat([df_features, df_cuisine_hot_encoding], axis=1)


random_forest_model = RandomForestClassifier(n_estimators=1000, max_depth=5)
random_forest_model.fit(X=df_features, y=df_raw_label['label'])

from sklearn import metrics
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(df_raw_label['label'], random_forest_model.predict(df_features), pos_label=1)
roc_auc = metrics.auc(fpr, tpr)


y_italian = random_forest_model.predict(df_features[df_features['cuisine_Italian']>0])

df_italian = df_raw_label[df_raw_label['cuisine'] == 'Italian']
df_italian['predicted_label'] = y_italian

df_italian[df_italian['label'] != df_italian['predicted_label']]




print(df_raw_label[df_raw_label['label'] == 1]['dish'].tolist())
print(df_raw_label[df_raw_label['label'] == 0]['dish'].tolist())

