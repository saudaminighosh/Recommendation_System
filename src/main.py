from preprocess import load_data
from popularity import recommend_popular
from collaborative import (
    create_user_item_matrix,
    compute_item_similarity,
    recommend_similar_products
)
from reorder_model import build_reorder_model


def main():

    print("Loading data...")
    orders, products, order_prior = load_data(
        r"C:\Users\sauda\OneDrive\Projects\Data Science\Recommendation_System\data"
    )

    print("\nTop Popular Products:")
    print(recommend_popular(order_prior, products, n=5))

    print("\nTop Reordered Products:")
    print(recommend_popular(order_prior, products, n=5, reorder_based=True))

    print("\nSampling smaller dataset for Collaborative Filtering...")

    # 🔹 Merge to get user_id inside order_prior
    order_prior_merged = order_prior.merge(
        orders[["order_id", "user_id"]],
        on="order_id",
        how="left"
    )

    # 🔹 Take first 3000 users
    sample_users = order_prior_merged["user_id"].unique()[:1000]

    order_prior_sample = order_prior_merged[
        order_prior_merged["user_id"].isin(sample_users)
    ]

    print("\nBuilding Collaborative Filtering...")
    user_item = create_user_item_matrix(order_prior_sample)
    similarity = compute_item_similarity(user_item)

    sample_product = products["product_id"].iloc[0]

    print("\nSimilar Products:")
    print(
        recommend_similar_products(
            sample_product,
            user_item,
            similarity,
            products,
            n=5
        )
    )

    print("\nTraining Reorder Prediction Model...")
    build_reorder_model(order_prior_sample)


if __name__ == "__main__":
    main()