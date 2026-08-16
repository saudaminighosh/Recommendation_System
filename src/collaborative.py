import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from src.db import engine
from sqlalchemy import text


def create_user_item_matrix():
    """
    Create a sparse user-product interaction matrix.

    PostgreSQL performs the expensive aggregation first,
    so Python does not need to process all raw purchase records.
    """

    query = """
        SELECT
            o.user_id,
            op.product_id,
            COUNT(*) AS purchase_count
        FROM orders o
        JOIN order_products op
            ON o.order_id = op.order_id
        GROUP BY
            o.user_id,
            op.product_id;
    """

    interactions = pd.read_sql(query, engine)

    # Map original IDs to matrix indices
    user_ids = interactions["user_id"].unique()
    product_ids = interactions["product_id"].unique()

    user_to_index = {
        user_id: index
        for index, user_id in enumerate(user_ids)
    }

    product_to_index = {
        product_id: index
        for index, product_id in enumerate(product_ids)
    }

    rows = interactions["user_id"].map(user_to_index)
    cols = interactions["product_id"].map(product_to_index)

    data = interactions["purchase_count"].astype(float)

    user_item_matrix = csr_matrix(
        (
            data,
            (rows, cols)
        ),
        shape=(len(user_ids), len(product_ids))
    )

    return user_item_matrix, user_ids, product_ids


def compute_item_similarity(user_item_matrix, n_neighbors=6):
    """
    Train a nearest-neighbor model using cosine distance.

    The model finds only the nearest products instead of
    creating a massive 49,677 x 49,677 similarity matrix.
    """

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors
    )

    # Transpose because we want product-product similarity
    model.fit(user_item_matrix.T)

    return model


def recommend_similar_products(
    product_id,
    user_item_matrix,
    product_ids,
    similarity_model,
    n=5
):
    """
    Recommend products similar to a given product.
    """

    product_id_to_index = {
        product_id: index
        for index, product_id in enumerate(product_ids)
    }

    if product_id not in product_id_to_index:
        return pd.DataFrame(columns=["product_name"])

    product_index = product_id_to_index[product_id]

    distances, indices = similarity_model.kneighbors(
        user_item_matrix.T[product_index],
        n_neighbors=n + 1
    )

    similar_indices = indices[0][1:]

    similar_product_ids = [
        product_ids[index]
        for index in similar_indices
    ]

    placeholders = ", ".join(
        ["%s"] * len(similar_product_ids)
    )

    query = f"""
        SELECT
            product_id,
            product_name
        FROM products
        WHERE product_id IN ({placeholders});
    """

    recommendations = pd.read_sql(
        query,
        engine,
        params=tuple(
            int(product_id)
            for product_id in similar_product_ids
        )
    )

    # Preserve similarity order
    recommendations["order"] = recommendations["product_id"].map(
        {
            product_id: index
            for index, product_id in enumerate(similar_product_ids)
        }
    )

    recommendations = recommendations.sort_values("order")

    return recommendations[
        ["product_id", "product_name"]
    ]


if __name__ == "__main__":

    print("Loading user-product interactions from PostgreSQL...")

    user_item_matrix, user_ids, product_ids = (
        create_user_item_matrix()
    )

    print(
        f"Users: {len(user_ids)}"
    )

    print(
        f"Products: {len(product_ids)}"
    )

    print(
        f"Non-zero interactions: "
        f"{user_item_matrix.nnz}"
    )

    print("\nTraining nearest-neighbor model...")

    similarity_model = compute_item_similarity(
        user_item_matrix,
        n_neighbors=6
    )

    # Test product
    test_product_id = 24852

    recommendations = recommend_similar_products(
        test_product_id,
        user_item_matrix,
        product_ids,
        similarity_model,
        n=5
    )

    print(
        f"\nProducts similar to product ID "
        f"{test_product_id}:"
    )

    print(recommendations)