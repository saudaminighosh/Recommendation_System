import pandas as pd
import os

def load_data(data_path):
    orders = pd.read_csv(os.path.join(data_path, "orders.csv"))
    products = pd.read_csv(os.path.join(data_path, "products.csv"))
    order_prior = pd.read_csv(os.path.join(data_path, "order_products__prior.csv"))
    
    return orders, products, order_prior


def merge_data(products, order_prior):
    merged = order_prior.merge(products, on="product_id")
    return merged