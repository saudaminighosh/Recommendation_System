from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def build_reorder_model(order_prior):
    
    product_stats = (
        order_prior.groupby("product_id")
        .agg({
            "reordered": "mean",
            "order_id": "count"
        })
        .reset_index()
    )
    
    product_stats.columns = ["product_id", 
                             "reorder_rate", 
                             "purchase_count"]
    
    X = product_stats[["purchase_count"]]
    y = product_stats["reorder_rate"] > 0.5
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    
    print("Reorder Prediction ROC-AUC:", auc)
    
    return model