import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Drop unused columns
    X = df.drop(columns=['datasetId', 'condition'])
    y = df['condition']

    # One-hot encode the labels
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_encoded = onehot_encoder.fit_transform(y.to_numpy().reshape(-1, 1))

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

    # Normalize features (fit scaler on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).reshape(-1, 1, X.shape[1])
    X_test_scaled = scaler.transform(X_test).reshape(-1, 1, X.shape[1])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, onehot_encoder
