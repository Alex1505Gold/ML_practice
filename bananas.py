import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns

def conversion(text):
    if text == 'Good':
        return 1
    return 0

def task(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for {model_name}: {accuracy}\n')

if __name__ == "__main__":
    data = 'banana_quality.csv'

    df = pd.read_csv(data)

    df = pd.read_csv('banana_quality.csv')
    df['Quality'] = df['Quality'].map(conversion)

    X = df.drop('Quality', axis=1)  # Признаки
    y = df['Quality']  # Целевая переменная

    features = list(set(df.columns) - set(['Quality']))   

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)

    for idx, feat in enumerate(features):
        sns.boxplot(x='Quality', y=feat, data=df, ax=axes[idx // 3, idx % 3])
        axes[idx // 3, idx % 3].set_xlabel('Quality')
        axes[idx // 3, idx % 3].set_ylabel(feat) 

    plt.savefig('banana.png')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f'Good bananas in y_train: {y_train.sum()}')
    print(f'Count of rows in y_train: {y_train.shape[0]}')

    estimators = [
        ('svm', svm.SVC()), 
        ('knn', KNeighborsClassifier()), 
        ('rf', RandomForestClassifier())
    ]

    modelClf = StackingClassifier(estimators=estimators, final_estimator = MLPClassifier())

    models = {
        "SVM"          : svm.SVC(),
        "KNN"          : KNeighborsClassifier(),
        "RandomForest" : RandomForestClassifier(),
        "MLP"          : MLPClassifier(),
        "Ansamble"     : modelClf
    }

    for model in models:
        task(X_train, X_test, y_train, y_test, models[model], model)