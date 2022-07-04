import os.path
import json
import simplejson
import pandas as pd


OS_PATH = '/Users/yafa/Dropbox/Library/DeepLearning/CS598_DataMining/yelp_dataset_challenge_academic_dataset/'

with open(os.path.join(OS_PATH, 'yelp_academic_dataset_review.json')) as f:
    dict_review = simplejson.load(f)
    

with open(os.path.join(OS_PATH, 'yelp_academic_dataset_business.json')) as f:
    dict_business = json.load(f)
    
df_review = pd.read_json()

df_business = pd.read_parquet(os.path.join(OS_PATH, 'business.parquet'))
