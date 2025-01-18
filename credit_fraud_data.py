import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import RandomOverSampler,SMOTE
from imblearn.under_sampling import  RandomUnderSampler
from collections import Counter
from imblearn.pipeline import Pipeline

def load_data(path):
    """
    Function to load data from a CSV file, preprocess it, and separate features and target.
    - Reads the CSV file into a DataFrame.
    - Prepares the data using the `prebare_data` function.
    - Converts the DataFrame to a NumPy array for further processing.
    - Separates the features (x_df) from the target (t_df).
    Returns: x_df (features) and t_df (target).
    """
    df = pd.read_csv(path)
    df = prebare_data(df)

    df = df.to_numpy()
    x_df = df[:, :-1]
    t_df = df[:, -1]

    return x_df, t_df


def prebare_data(data):
    """
    Function to preprocess data by applying logarithmic transformation to 'Amount' and 'Time' columns.
    - Applies log1p transformation to the 'Amount' column.
    - Applies log1p transformation to the 'Time' column.
    Returns: The transformed DataFrame.
    """
    data['Amount'] = np.log1p(data["Amount"])
    data['Time'] = np.log1p(data["Time"])

    return data


def Choose_procceros(option):
    """
    Function to choose a data processing scaler based on the given option.
    - If option is 1, returns MinMaxScaler.
    - If option is 2, returns StandardScaler.
    Returns: The chosen scaler object.
    """
    if option == 1:
        return MinMaxScaler()
    elif option == 2:
        return StandardScaler()


def transform_data(x_train, x_val, option):
    """
    Function to transform training and validation data using the chosen scaler.
    - Chooses the scaler using the `Choose_procceros` function based on the given option.
    - Fits the scaler on the training data and transforms both training and validation data.
    Returns: Transformed training (x_train) and validation (x_val) data.
    """
    Processor = Choose_procceros(option)
    x_train = Processor.fit_transform(x_train)
    x_val = Processor.transform(x_val)

    return x_train, x_val

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbpipeline

def Get_Sampler( X_resampled, y_resampled,option, ratio,size):
    size_of_postive=int(ratio*size)
    size_of_negative=int((1-ratio)*size)
    Sampler = {
        1: Pipeline(steps=[('sampler', RandomOverSampler(sampling_strategy={1: size_of_postive}, random_state=42))]),
        2: Pipeline(steps=[('sampler', RandomUnderSampler(sampling_strategy={0: size_of_negative}, random_state=42))]),
        3: Pipeline(steps=[
            ('under_sampler', RandomUnderSampler(sampling_strategy={0: size_of_negative}, random_state=42)),
            ('over_sampler', RandomOverSampler(sampling_strategy={1: size_of_postive}, random_state=42))
        ])
    }

    X_resampled, y_resampled = Sampler[option].fit_resample(X_resampled, y_resampled)
    return X_resampled ,y_resampled





