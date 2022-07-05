import os
import ast
import os
import re

import pandas as pd

from CS598_DataMining.io_embedding import HOME_PATH


def read_data(data_folder=os.path.join(HOME_PATH, 'yelp_dataset_challenge_academic_dataset/YelpHygieneData/')):
    # from CS598_DataMining.io_embedding import read_file_with_embedding, RAW_DATA_PATH, HOME_PATH
    
    with open(os.path.join(data_folder, 'hygiene.dat')) as f:
        hygiene_raw = f.readlines()
    
    with open(os.path.join(data_folder, 'hygiene.dat.labels')) as f:
        hygiene_labels = f.readlines()
    
    with open(os.path.join(data_folder, 'hygiene.dat.additional')) as f:
        hygiene_additional = f.readlines()
    
    for index_i, i in enumerate(hygiene_additional):
        i_ = ast.literal_eval('(' + i + ')')
        i_ = list(i_)
        i_[0] = ast.literal_eval(i_[0])
        
        hygiene_additional[index_i] = i_
    
    df_hygiene_additional = pd.DataFrame(hygiene_additional)
    df_hygiene_additional.columns = ['cusine', 'zipcode', 'review_count', 'avg_star']
    
    list_hygiene_labels = []
    for index_i, i in enumerate(hygiene_labels):
        # index_i = 0; i = hygiene_labels[index_i]
        d = re.compile(r'\d+').findall(i)
        if d:
            list_hygiene_labels.append(int(d[0]))
        else:
            list_hygiene_labels.append(None)
    
    for index_i, i in enumerate(hygiene_raw):
        hygiene_raw[index_i] = i.strip()
    
    df_hygiene = pd.DataFrame({'review': hygiene_raw, 'label': list_hygiene_labels})
    
    df_hygiene = pd.concat([df_hygiene, df_hygiene_additional], axis=1)
    
    df_hygiene_train = df_hygiene[~df_hygiene['label'].isna()]
    df_hygiene_test = df_hygiene[df_hygiene['label'].isna()]
    
    return df_hygiene_train, df_hygiene_test
