from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def model_evaluation(y, y_pred, split, tune):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    result = pd.DataFrame({
        "score": [accuracy, precision, recall, f1],
        "metric": ["Accuracy", "Precision", "Recall", "F1"],
        "split": split,
        "tune": tune})

    return result



