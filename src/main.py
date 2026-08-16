from src.popularity import recommend_popular

from src.collaborative import (
    create_user_item_matrix,
    compute_item_similarity,
    recommend_similar_products
)

from src.reorder_model import build_reorder_model


def main():

    print("=" * 60)
    print("GROCERY RECOMMENDATION SYSTEM")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Popularity-based recommendations
    # ---------------------------------------------------------

    print("\n1. TOP POPULAR PRODUCTS")
    print("-" * 40)

    popular_products = recommend_popular(
        n=5,
        reorder_based=False
    )

    print(popular_products)


    # ---------------------------------------------------------
    # 2. Reorder-based recommendations
    # ---------------------------------------------------------

    print("\n2. TOP REORDERED PRODUCTS")
    print("-" * 40)

    reordered_products = recommend_popular(
        n=5,
        reorder_based=True
    )

    print(reordered_products)


    # ---------------------------------------------------------
    # 3. Collaborative filtering
    # ---------------------------------------------------------

    print("\n3. COLLABORATIVE FILTERING")
    print("-" * 40)

    print("Loading user-product interactions from PostgreSQL...")

    user_item_matrix, user_ids, product_ids = (
        create_user_item_matrix()
    )

    print(f"Users: {len(user_ids)}")
    print(f"Products: {len(product_ids)}")
    print(
        f"Non-zero interactions: "
        f"{user_item_matrix.nnz}"
    )

    print("\nTraining nearest-neighbor model...")

    similarity_model = compute_item_similarity(
        user_item_matrix,
        n_neighbors=6
    )

    # Banana product
    sample_product = 24852

    print(
        f"\nProducts similar to product "
        f"{sample_product}:"
    )

    similar_products = recommend_similar_products(
        sample_product,
        user_item_matrix,
        product_ids,
        similarity_model,
        n=5
    )

    print(similar_products)


    # ---------------------------------------------------------
    # 4. Reorder prediction model
    # ---------------------------------------------------------

    print("\n4. REORDER PREDICTION MODEL")
    print("-" * 40)

    model = build_reorder_model()

    print("\nReorder prediction model trained successfully.")


    print("\n" + "=" * 60)
    print("RECOMMENDATION SYSTEM COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()