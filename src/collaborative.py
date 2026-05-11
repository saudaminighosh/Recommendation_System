import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def create_user_item_matrix(order_prior):
    user_item = pd.crosstab(order_prior["order_id"], 
                             order_prior["product_id"])
    return user_item

def compute_item_similarity(user_item_matrix):
    item_similarity = cosine_similarity(user_item_matrix.T)
    return item_similarity

def recommend_similar_products(product_id, user_item_matrix, 
                                similarity_matrix, products, n=5):
    
    product_index = list(user_item_matrix.columns).index(product_id)
    
    similarity_scores = similarity_matrix[product_index]
    
    similar_indices = similarity_scores.argsort()[::-1][1:n+1]
    
    similar_product_ids = user_item_matrix.columns[similar_indices]
    
    recommendations = products[
        products["product_id"].isin(similar_product_ids)
    ]
    
    return recommendations[["product_name"]]