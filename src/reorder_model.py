import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.db import engine


def build_reorder_model():

    query = """
        SELECT
            product_id,
            AVG(reordered) AS reorder_rate,
            COUNT(order_id) AS purchase_count
        FROM order_products
        GROUP BY product_id;
    """

    product_stats = pd.read_sql(
        query,
        engine
    )

    # Create binary target
    # True  = reorder rate > 50%
    # False = reorder rate <= 50%
    X = product_stats[["purchase_count"]]

    y = product_stats["reorder_rate"] > 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression()

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        preds
    )

    print(
        "Reorder Prediction ROC-AUC:",
        auc
    )

    return model


if __name__ == "__main__":

    model = build_reorder_model()

    print("\nReorder model trained successfully.")