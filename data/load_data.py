import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

orders = pd.read_csv('/mnt/c/Users/sauda/OneDrive/Projects/Data Science/Recommendation_System/data/orders.csv')
products = pd.read_csv('/mnt/c/Users/sauda/OneDrive/Projects/Data Science/Recommendation_System/data/products.csv')
order_prior = pd.read_csv('/mnt/c/Users/sauda/OneDrive/Projects/Data Science/Recommendation_System/data/order_products__prior.csv')
departments = pd.read_csv('/mnt/c/Users/sauda/OneDrive/Projects/Data Science/Recommendation_System/data/departments.csv')
aisles = pd.read_csv('/mnt/c/Users/sauda/OneDrive/Projects/Data Science/Recommendation_System/data/aisles.csv')
print("Orders:", orders.shape)
print("Products:", products.shape)
print("Order-Prior:", order_prior.shape)
print("Departments:", departments.shape)
print("Aisles:", aisles.shape)
a=order_prior['reordered'].value_counts(normalize=True)
top_reordered = (
    order_prior.groupby('product_id')['reordered']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_reordered)
top_reordered = top_reordered.reset_index()
top_reordered = top_reordered.merge(products, on='product_id')
top_reordered[['product_name', 'reordered']]
merged = order_prior.merge(products, on='product_id')
merged = merged.merge(departments, on='department_id')
dept_popularity = merged['department'].value_counts().head(10)
#print(dept_popularity)
op_reordered = top_reordered.reset_index()
top_reordered = top_reordered.merge(products, on='product_id')
top_reordered[['product_name', 'reordered']]
merged = order_prior.merge(products, on='product_id')
merged = merged.merge(departments, on='department_id')
dept_popularity = merged['department'].value_counts().head(10)
print(dept_popularity)
