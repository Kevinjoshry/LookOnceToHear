from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, Flatten
from data_preprocessing import preprocess_data
import pandas as pd

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same', input_shape=input_shape))
    model.add(Dense(units=16, activation='relu'))
    model.add(MaxPool1D(pool_size=1, strides=1))
    model.add(Flatten())
    model.add(Dense(units=num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_save_model(df, save_path="model.h5"):
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data(df)

    model = build_model(input_shape=(1, X_train.shape[2]), num_classes=y_train.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.05)

    # Save model
    model.save(save_path)

    # Save scaler and encoder if needed (using joblib or pickle)
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")

    return model, history
