# Neural-Network-for-Multiclass-classification
# Importing Libraries
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as 
```
# Loading Dataset
```python
df = pd.read_csv('/content/train.csv')

df.head()

df.info()
```
#Encoding Attributes to Numeric value
```python#
def prepareFeatures(df):
    # Encode string features to numerics
    columns = df.select_dtypes(include=['object']).columns
    # columns = ["Gender","Ever_Married","Graduated","Profession","Spending_Score"]
    for feature in columns:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    # fill in missing features with mean values
    features_missing = df.columns[df.isna().any()]
    for feature in features_missing:
        fillmissing(df, feature=feature, method="mean")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(df)
 ```
 # Trainin with batch of 100 and 200 epochs
 ```python
  model.fit(X, hot_y, validation_split=0.33,
                    epochs=200, batch_size=100, verbose=0)
```
