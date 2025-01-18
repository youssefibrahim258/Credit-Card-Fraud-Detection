import argparse
from credit_fraud_data import load_data ,Choose_procceros,transform_data,Get_Sampler
from credit_fraud_model import Get_model,predict_eval,test_eval,find_best_config,find_best_threshold_using_prc
import pickle
import warnings
warnings.filterwarnings('ignore')
import json

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description="Credit Card Fraud Detection")

    parser.add_argument("--train_bath",type=str,default=r"Y:\01 ML\Projects\03 credit card\split\train.csv",
                        help="train_bath")

    parser.add_argument("--val_bath",type=str,default=r"Y:\01 ML\Projects\03 credit card\split\val.csv",
                        help="val_bath")

    parser.add_argument("--preprocessor_options",type=int, nargs="+", default=2,
                        help="1 for MinMaxScaler() and 2 for StandardScaler()")

    parser.add_argument("--model_options", type=int, nargs="+", default=[1, 2, 3, 4], help=
    '''1 for LogisticRegression, 2 for RandomForestClassifier,
     3 for VotingClassifier (Logistic + RandomForest), 4 for Neural Network Classifier (MLP)''')

    parser.add_argument('--sampler_options', type=int, nargs="+", default=[1, 2], help=
    '''
    1 for OverSampling 
    2 for UnderSampling 
    3 for UnderSampling followed by OverSampling 
    ''')

    args=parser.parse_args()

    #load_data
    x_train,t_train=load_data(args.train_bath)
    x_val,t_val=load_data(args.val_bath)

    """
    Defines hyperparameter grids for different models to be used in RandomizedSearchCV, covering aspects like regularization, tree settings, model weights, and activation functions.
    """
    model_grids = {
        1: {  # Logistic Regression
            'C': [0.01, 0.1, 1, 10],
            "max_iter": [50, 100, 150, 200],
            'solver': ["lbfgs"],
            'class_weight': [
                {0: 0.1, 1: 1},
                {0: 0.5, 1: 5},
                {0: 0.2, 1: 1}
            ]
        },
        2: {  # Random Forest
            'n_estimators': [50, 100, 200],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            "max_depth": [10, 20, 30],
            'class_weight': [
                {0: 0.1, 1: 1},
                {0: 0.2, 1: 10},
                {0: 0.5, 1: 5}
            ]
        },
        3: {  # Voting Classifier
            'weights': [[1, 1], [2, 1], [3, 1]],
            'voting': ['soft']
        },
        4: {  # MLP
            'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [200, 300, 500],
            'solver': ['adam']
        }
    }

    """
    Filters the model grids to include only the selected models from the command-line arguments and then finds the best configuration by evaluating different sampler options, model options, and preprocessor options using the find_best_config function.
    """
    selected_models = args.model_options
    model_grids = {k: v for k, v in model_grids.items() if k in selected_models}
    # Find best configuration
    best_config = find_best_config(
        train_data=(x_train, t_train),
        val_data=(x_val, t_val),
        sampler_options=args.sampler_options,
        model_options=model_grids,
        preprocessor_options=args.preprocessor_options,
    )



    """
    Loads the best model configurations from a JSON file, searches for the best RandomForestClassifier configuration, extracts its parameters and best threshold, and modifies the class_weight dictionary keys from strings to integers.
    """
    with open('best_configurations.json', 'r') as file:
        data = json.load(file)

    best_params = None
    best_sampler=None
    model='RandomForestClassifier'
    for config in data['best_configurations']:
        if config['model'] == model:
            best_params = config['params']
            best_threshold=config['best_threshold']
            best_sampler=config['sampler']
            break

    if model in ['LogisticRegression', 'RandomForestClassifier']:
        if isinstance(best_params['class_weight'], dict):
            best_params['class_weight'] = {int(k): v for k, v in best_params['class_weight'].items()}


    """
    Applies the selected preprocessor to the training and validation data, then uses the chosen sampler to balance the class distribution in the training data. If the sampler includes both undersampling and oversampling, applies both in sequence.
    """
    preprocessor = Choose_procceros(args.preprocessor_options)
    X_train = preprocessor.fit_transform(x_train)
    X_val = preprocessor.transform(x_val)

    size = len(t_train[t_train == 0])
    ratio=0.02
    # apply best sampler
    if best_sampler =='OverSampling':
        X_train_resampled, T_train_resampled = Get_Sampler(X_train,t_train,args.sampler_options[0],ratio,size)
    elif best_sampler =='UnderSampling' :
        X_train_resampled, T_train_resampled = Get_Sampler(X_train,t_train,args.sampler_options[1],ratio,size)
    elif best_sampler =='Undersampling then Oversampling':
        X_train_resampled, T_train_resampled = Get_Sampler(X_train,t_train,args.sampler_options[2],ratio,size)

    print(len(X_train_resampled))


    """
    Creates and trains the selected model using the best hyperparameters for RandomForest, then evaluates it on both the training and validation datasets using the best threshold. The test_eval function is used to calculate and print the precision, recall, and F1 score for both datasets.
    """
    if model == 'LogisticRegression':
        model = Get_model(args.model_options[0],**best_params)
    elif model == 'RandomForestClassifier':
        model = Get_model(args.model_options[1],**best_params)
    elif model =='VotingClassifier':
        model = Get_model(args.model_options[2],**best_params)
    elif model =='MLPClassifier':
        model = Get_model(args.model_options[3],**best_params)


    model.fit(X_train_resampled, T_train_resampled)


    predict_eval(model,X_train_resampled,T_train_resampled,best_threshold)

    test_eval(model,X_val,t_val,best_threshold)

    """
    Saves the trained model and its best threshold in a dictionary and serializes it to a file named "model.pkl" using pickle for later use.
    """
    model_dict={
        "model":model ,
        "best_threshold":best_threshold
    }
    filename = "RandomForestClassifier.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model_dict, file)















