from sklearn.linear_model import LogisticRegression

def get_model_v01() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000
    )