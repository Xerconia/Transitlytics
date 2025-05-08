  # Scripts for training machine learning models

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    # Example preprocessing steps
    df.fillna(method='ffill', inplace=True)  # Forward fill for missing values
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding for categorical variables
    return df

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # Load the data
    data = load_data('data/processed/trip_data.csv')  # Adjust the path as necessary

    # Preprocess the data
    processed_data = preprocess_data(data)

    # Split the data into features and target
    X = processed_data.drop('trip_purpose', axis=1)  # Features
    y = processed_data['trip_purpose']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    # Save the model
    joblib.dump(model, 'models/trip_purpose_model.pkl')

if __name__ == "__main__":
    main()