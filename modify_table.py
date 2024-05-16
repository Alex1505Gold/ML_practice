import pandas as pd

def conversion(text):
    if text == 'Good':
        return 1
    else:
        return 0
    

df = pd.read_csv('banana_quality.csv')
df['Quality'] = df['Quality'].map(conversion)
print(df)