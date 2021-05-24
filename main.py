import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

FORCAST_DAYS = 5
TEST_SIZE = 0.9

df = pd.read_csv('./bitcoin_clean.csv')
print(f'Initial length of the columns: {len(df)}.')

# Creating a label column for the prediction by moving the Close column 
# backwards by the desired amount of forcast days. In this way the feature columns will
# point to the close price in the future.

df['Label'] = df['Close'].shift(-FORCAST_DAYS)
df.dropna(inplace=True)

# We're creating a features array that doesn't include the time and 
# label and then we're seperating it into the rows that we want to train 
# and into the rows that we'll use for prediction.

FEATURES = preprocessing.scale(np.array(df.drop(['Label', 'Timestamp'], axis=1)))
features = FEATURES[:-FORCAST_DAYS]
features_after = FEATURES[-FORCAST_DAYS:]

# Creating a label array for training without the days that will be predicted.

label = np.array(df['Label'])
label = label[:-FORCAST_DAYS]
print(len(label), len(features))

# The previous data is being prepared for training and testing.

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=TEST_SIZE)

# Creating a linear regression classifier, training and testing it. 

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(f'Accuracy: {accuracy}.')

# Using the classifier to predict the data and comparing it to the real data.

forcast_set = clf.predict(features_after)
print(forcast_set)
print(df.tail(FORCAST_DAYS + 1))