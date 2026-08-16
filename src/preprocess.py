import pandas as pd
from src.db import engine


def load_orders():
    """
    Load order information from PostgreSQL.
    """

    query = """
        SELECT
            order_id,
            user_id,
            eval_set,
            order_number,
            order_dow,
            order_hour_of_day,
            days_since_prior_order
        FROM orders;
    """

    return pd.read_sql(query, engine)


def load_products():
    """
    Load product information from PostgreSQL.
    """

    query = """
        SELECT
            product_id,
            product_name,
            aisle_id,
            department_id
        FROM products;
    """

    return pd.read_sql(query, engine)


def load_user_product_history(user_id):
    """
    Retrieve purchase history for a specific user.
    Filtering is performed directly in PostgreSQL.
    """

    query = """
        SELECT
            op.order_id,
            op.product_id,
            op.add_to_cart_order,
            op.reordered,
            p.product_name,
            p.aisle_id,
            p.department_id,
            o.user_id,
            o.order_number,
            o.order_dow,
            o.order_hour_of_day,
            o.days_since_prior_order
        FROM order_products op
        JOIN products p
            ON op.product_id = p.product_id
        JOIN orders o
            ON op.order_id = o.order_id
        WHERE o.user_id = %(user_id)s;
    """

    return pd.read_sql(
        query,
        engine,
        params={"user_id": user_id}
    )


if __name__ == "__main__":
    orders = load_orders()

    print("Orders loaded:", len(orders))
    print(orders.head())