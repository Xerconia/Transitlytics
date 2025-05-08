  # Utility functions
def load_data(file_path):
    import pandas as pd
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Save DataFrame to a CSV file."""
    data.to_csv(file_path, index=False)

def encode_categorical(data, columns):
    """Convert categorical columns to numerical using one-hot encoding."""
    return pd.get_dummies(data, columns=columns, drop_first=True)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using accuracy and other metrics."""
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report