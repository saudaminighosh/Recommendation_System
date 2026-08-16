import pandas as pd
from src.db import engine


def recommend_popular(n=10, reorder_based=False):

    if reorder_based:

        query = """
            SELECT
                op.product_id,
                p.product_name,
                SUM(op.reordered) AS score
            FROM order_products op
            JOIN products p
                ON op.product_id = p.product_id
            GROUP BY op.product_id, p.product_name
            ORDER BY score DESC
            LIMIT %(n)s;
        """

    else:

        query = """
            SELECT
                op.product_id,
                p.product_name,
                COUNT(*) AS score
            FROM order_products op
            JOIN products p
                ON op.product_id = p.product_id
            GROUP BY op.product_id, p.product_name
            ORDER BY score DESC
            LIMIT %(n)s;
        """

    top_n = pd.read_sql(
        query,
        engine,
        params={"n": n}
    )

    return top_n[["product_name", "score"]]


if __name__ == "__main__":
    recommendations = recommend_popular(n=10)

    print("\nTop 10 Popular Products:")
    print(recommendations)