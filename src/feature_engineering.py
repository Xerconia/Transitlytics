  # Scripts for feature extraction

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def extract_features(data):
    """
    Extract features from the dataset for trip purpose prediction.

    Parameters:
    data (DataFrame): The input DataFrame containing household and individual characteristics.

    Returns:
    DataFrame: A DataFrame containing the extracted features.
    """
    # Example feature extraction
    features = pd.DataFrame()

    # One-hot encoding for categorical variables
    encoder = OneHotEncoder(sparse=False, drop='first')
    categorical_features = encoder.fit_transform(data[['URBAN', 'VEHTYPE', 'SEX']])
    feature_names = encoder.get_feature_names_out(['URBAN', 'VEHTYPE', 'SEX'])
    
    features = pd.DataFrame(categorical_features, columns=feature_names)

    # Adding numerical features
    features['HHFAMINC'] = data['HHFAMINC']
    features['HHSIZE'] = data['HHSIZE']
    features['AGE'] = data['AGE']
    
    return features

def prepare_data(data):
    """
    Prepare the data for modeling by extracting features and splitting the dataset.

    Parameters:
    data (DataFrame): The input DataFrame containing household and individual characteristics.

    Returns:
    tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """
    # Extract features
    X = extract_features(data)
    
    # Assuming 'trip_purpose' is the target variable
    y = data['trip_purpose']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test