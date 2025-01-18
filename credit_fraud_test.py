import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def load_data(path):
    df = pd.read_csv(path)
    df = prebare_data(df)

    df = df.to_numpy()
    x_df = df[:, :-1]
    t_df = df[:, -1]

    return x_df, t_df


def prebare_data(data):
    data['Amount'] = np.log1p(data["Amount"])
    data['Time'] = np.log1p(data["Time"])

    return data

def test_eval(model, X_test, y_test, train_threshold):
    probabilities = model.predict_proba(X_test)[:, 1]

    if not isinstance(train_threshold, (float, int)):
        raise ValueError("train_threshold must be a single numerical value.")

    y_pred = (probabilities >= train_threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision on Test Data = {precision:.4f}")
    print(f"Recall on Test Data    = {recall:.4f}")
    print(f"F1-Score on Test Data  = {f1:.4f}")

if __name__=='__main__':
    path_test=r"Y:\01 ML\Projects\03 credit card\split\test.csv"
    X_test,t_test=load_data(path_test)

    filename = "RandomForestClassifier.pkl"

    with open(filename, "rb") as file:
        loaded_model_dict = pickle.load(file)

    best_threshold = loaded_model_dict.get("best_threshold", None)
    model=loaded_model_dict.get("model",None)

    test_eval(model,X_test,t_test,best_threshold)


