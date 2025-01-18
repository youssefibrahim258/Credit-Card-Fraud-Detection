import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from credit_fraud_data import Choose_procceros,Get_Sampler


def Get_model(option,**hyperparameters):
    """
    Function to get a machine learning model based on the specified option and hyperparameters.
    - If option is 1: Returns a LogisticRegression model with the given hyperparameters.
    - If option is 2: Returns a RandomForestClassifier model with the given hyperparameters.
    - If option is 3: Returns a VotingClassifier combining LogisticRegression and RandomForestClassifier models using soft voting.
    - If option is 4: Returns an MLPClassifier model with the given hyperparameters.
    Returns: The chosen machine learning model.
    """

    if option == 1:
        model = LogisticRegression(**hyperparameters)
    elif option == 2:
        model = RandomForestClassifier(**hyperparameters)
    elif option == 3:
        # Define two models for the VotingClassifier
        model1 = LogisticRegression()
        model2 = RandomForestClassifier()
        model = VotingClassifier(estimators=[('logistic', model1), ('random_forest', model2)],**hyperparameters)
    elif option == 4:
        model = MLPClassifier(**hyperparameters)

    return model

def find_best_threshold_using_prc(model, X_train, t_train):
    """ Find the best threshold based on the Precision-Recall Curve using training data. """
    y_probs = model.predict_proba(X_train)[:, 1]  # Get probabilities for class 1
    precision, recall, thresholds = precision_recall_curve(t_train, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)  # Calculate F1 score for each threshold
    best_threshold = thresholds[np.argmax(f1_scores)]  # Find the threshold that maximizes F1 score
    return best_threshold



def find_best_config(train_data, val_data, sampler_options, model_options, preprocessor_options, random_state=42, output_file="best_configurations.json"):
    """
    Function to find the best configuration for each model based on F1 score, comparing Oversampling and Undersampling.
    - Evaluates different sampling techniques and selects the best configuration for each model.
    - Saves the best configurations to a JSON file.
    Returns: The best configurations for each model.
    """
    x_train, t_train = train_data
    x_val, t_val = val_data

    best_model_configs = {}

    model_names = {
        1: 'LogisticRegression',
        2: 'RandomForestClassifier',
        3: 'VotingClassifier',
        4: 'MLPClassifier',
    }

    preprocessor_names = {
        1: 'MinMaxScaler',
        2: 'StandardScaler'
    }

    sampler_names = {
        1: 'OverSampling',
        2: 'UnderSampling',
    }

    # Apply preprocessor
    preprocessor = Choose_procceros(preprocessor_options)
    X_train = preprocessor.fit_transform(x_train)
    X_val = preprocessor.transform(x_val)

    size = len(t_train[t_train == 0])
    ratio = 0.1
    for sampler_option in sampler_options:

        if sampler_option == 1:
            X_train_resampled, T_train_resampled = Get_Sampler(X_train, t_train, sampler_option,ratio,size)
        elif sampler_option == 2:
            X_train_resampled, T_train_resampled = Get_Sampler(X_train, t_train, sampler_option,ratio,size)
        elif sampler_option == 3:
            X_train_resampled, T_train_resampled = Get_Sampler(X_train, t_train, sampler_option,ratio,size)



        for model_option, param_distributions in model_options.items():
            model = Get_model(model_option)

            # Remove random_state for VotingClassifier
            if model.__class__.__name__ == 'VotingClassifier':
                if 'random_state' in param_distributions:
                    del param_distributions['random_state']
            else:
                param_distributions['random_state'] = [random_state]

            # Initialize RandomizedSearchCV
            stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            random_search = RandomizedSearchCV(
                model, param_distributions, n_iter=20, scoring='f1', cv=stratified_kfold,
                n_jobs=-1, verbose=2, random_state=random_state
            )

            random_search.fit(X_train_resampled, T_train_resampled)

            best_params = random_search.best_params_
            model_rf = random_search.best_estimator_

            # Find the best threshold using Precision-Recall curve
            best_threshold = find_best_threshold_using_prc(model_rf, X_train_resampled, T_train_resampled)

            # Calculate F1 scores
            y_probs_train = model_rf.predict_proba(X_train_resampled)[:, 1]
            y_pred_train = (y_probs_train >= best_threshold).astype(int)
            train_f1 = f1_score(T_train_resampled, y_pred_train)

            y_probs_val = model_rf.predict_proba(X_val)[:, 1]
            y_pred_val = (y_probs_val >= best_threshold).astype(int)
            score_val_f1 = f1_score(t_val, y_pred_val)

            # Current configuration
            current_config = {
                "model": model_names[model_option],
                "preprocessor": preprocessor_names[preprocessor_options],
                "sampler": sampler_names[sampler_option],
                "params": best_params,
                "train_f1_score": train_f1,
                "val_score": score_val_f1,
                "best_threshold": best_threshold,
            }

            # Update the best configuration for the model
            model_name = model_names[model_option]
            if model_name not in best_model_configs or score_val_f1 > best_model_configs[model_name]["val_score"]:
                best_model_configs[model_name] = current_config
                print(f"\nUpdated Best Configuration for {model_name}:")
                print(current_config)

    # Convert the best configurations to a list
    final_best_config_list = list(best_model_configs.values())

    # Save the best configurations to a JSON file
    with open(output_file, "w") as file:
        json.dump({"best_configurations": final_best_config_list}, file, indent=4)

    print("\nFinal Best Configurations for Each Model:")
    for config in final_best_config_list:
        print(config)

    return final_best_config_list



def predict_eval(model, x_train, t_train,best_threshold):
    """
    Function to evaluate the model's performance on training data using the given threshold.
    - Calculates the predicted probabilities for class 1 on the training data.
    - Ensures the provided threshold is a numerical value.
    - Converts probabilities to binary predictions using the provided threshold.
    - Computes precision, recall, and F1-score based on the training data.
    - Prints precision, recall, and F1-score on the training data.
    """
    probabilities = model.predict_proba(x_train)[:, 1]

    if not isinstance(best_threshold, (float, int)):
        raise ValueError("train_threshold must be a single numerical value.")

    y_pred = (probabilities >= best_threshold).astype(int)

    precision = precision_score(t_train, y_pred)
    recall = recall_score(t_train, y_pred)
    f1 = f1_score(t_train, y_pred)

    print(f"Precision on Train Data = {precision:.4f}")
    print(f"Recall on Train Data    = {recall:.4f}")
    print(f"F1-Score on Train Data  = {f1:.4f}")



def test_eval(model, X_val, t_val, train_threshold):
    """
    Function to evaluate the model's performance on validation data using the given threshold.
    - Calculates the predicted probabilities for class 1 on the validation data.
    - Ensures the provided threshold is a numerical value.
    - Converts probabilities to binary predictions using the provided threshold.
    - Computes precision, recall, and F1-score based on the validation data.
    - Prints precision, recall, and F1-score on the validation data.
    """
    probabilities = model.predict_proba(X_val)[:, 1]

    if not isinstance(train_threshold, (float, int)):
        raise ValueError("train_threshold must be a single numerical value.")

    y_pred = (probabilities >= train_threshold).astype(int)

    precision = precision_score(t_val, y_pred)
    recall = recall_score(t_val, y_pred)
    f1 = f1_score(t_val, y_pred)

    print(f"Precision on Test Data = {precision:.4f}")
    print(f"Recall on Test Data    = {recall:.4f}")
    print(f"F1-Score on Test Data  = {f1:.4f}")









