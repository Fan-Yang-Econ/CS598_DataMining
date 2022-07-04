import json
import random

import pandas as pd

OS_PATH = '/yelp_dataset_challenge_academic_dataset/'
local_path_json_business = OS_PATH + "yelp_academic_dataset_business.json"
local_path_json_reviews = OS_PATH + "yelp_academic_dataset_review.json"


def clean_json_data_and_output_to_csv(sample_rate = 0.2):
    """
    save_sample = True
    save_categories = True
    
    :param save_sample:
    :param save_categories:
    :return:
    """
    categories = set([])
    restaurant_ids = set([])
    # For each cuisine category, list the restaurants
    category_2_restaurant = {}
    restaurant_2_reviewId = {}
    
    type_filter = 'Restaurants'
    with open(local_path_json_business, 'r') as f:
        for line in f.readlines():
            business_json = json.loads(line)
            bjc = business_json['categories']
            # cities.add(business_json['city'])
            if type_filter in bjc:
                if len(bjc) > 1:
                    # print(bjc)
                    restaurant_ids.add(business_json['business_id'])
                    categories = set(bjc).union(categories) - set([type_filter])
                    stars = business_json['stars']
                    for cat in bjc:
                        if cat == type_filter:
                            continue
                        if cat in category_2_restaurant:
                            category_2_restaurant[cat].append(business_json['business_id'])
                        else:
                            category_2_restaurant[cat] = [business_json['business_id']]
                        
    
    nz_count = 0
    
    # A category is valid only if it gets 30+ reviews.
    VALID_CATEGORY = []
    for i, cat in enumerate(category_2_restaurant):
        cat_total_reviews = 0
        for rid in category_2_restaurant[cat]:
            # number of reviews for each of restaurants
            if rid in restaurant_2_reviewId:
                cat_total_reviews = cat_total_reviews + len(restaurant_2_reviewId[rid])
        
        if cat_total_reviews > 30:
            nz_count = nz_count + 1
            VALID_CATEGORY.append(cat)
            # print( cat, cat_total_reviews)
    
    df_business = []
    random.seed(1)
    type_filter = 'Restaurants'
    with open(local_path_json_business, 'r') as f:
        for line in f.readlines():
            business_json = json.loads(line)
            bjc = business_json['categories']
            # cities.add(business_json['city'])
            if len(bjc) > 1 and type_filter in bjc:
                df_business.append({
                    'restaurant_id': business_json['business_id'],
                    'Price Range': business_json['attributes'].get('Price Range'),
                    'stars': business_json['stars'],
                    'categories': business_json['categories']
                })
    df_business = pd.DataFrame(df_business)
    
    random.seed(1)
    df_review = []
    with open(local_path_json_reviews, 'r') as f:
        for line in f.readlines():
            # random sample by 20%
            if random.randint(0, 100) < sample_rate * 100:
                review_json = json.loads(line)
                rid = review_json['business_id']
                if rid in df_business['restaurant_id'].tolist():
                    df_review.append(review_json)
    
    df_review = pd.DataFrame(df_review)
    del df_review['type']
    df_review['votes_useful'] = df_review['votes'].apply(lambda x: x['useful'])
    df_review['votes_funny'] = df_review['votes'].apply(lambda x: x['funny'])
    df_review['votes_cool'] = df_review['votes'].apply(lambda x: x['cool'])
    del df_review['votes']
    
    df_review.to_csv(OS_PATH + 'df_review', index_label=False)
    df_business.to_parquet(OS_PATH + 'df_business.parquet')


