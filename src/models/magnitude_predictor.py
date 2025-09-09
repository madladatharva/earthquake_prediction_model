import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

def feature_engineering(data):
    # Example feature engineering
    data['depth_squared'] = data['depth'] ** 2
    data['location_encoded'] = data['location'].astype('category').cat.codes
    return data

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    return model

def predict_rf(model, X):
    return model.predict(X)

def predict_nn(model, X):
    return model.predict(X)

def main(data):
    # Feature engineering
    data = feature_engineering(data)
    
    # Define features and target
    X = data[['location_encoded', 'depth', 'depth_squared']]
    y = data['magnitude']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train models
    rf_model = train_random_forest(X_train, y_train)
    nn_model = train_neural_network(X_train, y_train)

    # Predictions
    rf_predictions = predict_rf(rf_model, X_test)
    nn_predictions = predict_nn(nn_model, X_test)

    # Evaluate models
    rf_mse = mean_squared_error(y_test, rf_predictions)
    nn_mse = mean_squared_error(y_test, nn_predictions)

    print(f"Random Forest MSE: {rf_mse}")
    print(f"Neural Network MSE: {nn_mse}")

# Example usage (assuming 'data' is a DataFrame with appropriate columns)
# if __name__ == "__main__":
#     data = pd.read_csv('path_to_your_data.csv')
#     main(data)