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
