def recommend_popular_products(n=10, reorder_based=False):
    if reorder_based:
        popularity = (
            order_prior.groupby('product_id')['reordered']
            .sum()
            .sort_values(ascending=False)
        )
    else:
        popularity = (
            order_prior.groupby('product_id')
            .size()
            .sort_values(ascending=False)
        )
    
    top_n = popularity.head(n).reset_index()
    top_n.columns = ['product_id', 'score']
    
    top_n = top_n.merge(products, on='product_id')
    return top_n[['product_name', 'score']]
