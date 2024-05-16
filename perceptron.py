#from tensorflow.keras.models import Sequential
import keras
from keras import layers
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from bananas import conversion


if __name__ == "__main__":

    data = 'banana_quality.csv'

    df = pd.read_csv(data)

    df = pd.read_csv('banana_quality.csv')
    df['Quality'] = df['Quality'].map(conversion)

    X = df.drop('Quality', axis=1)  # Признаки
    y = df['Quality']  # Целевая переменная

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = keras.Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('banana_loss.png')

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int)
    print(classification_report(y_test, y_pred_bin))

    #temp = sum(y_test == y_pred)
    #print(temp/len(y_test))
'''
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}\n')

'''
