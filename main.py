import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

FORCAST_DAYS = 1
TEST_SIZE = 1

df = pd.read_csv('./bitcoin_clean.csv')
print(f'Initial length of the columns: {len(df)}.')

# Creating a label column for the prediction by moving the Close column 
# backwards by the desired amount of forcast days. In this way the feature columns will
# point to the close price in the future.

df['Label'] = df['Close'].shift(-FORCAST_DAYS)
df.dropna(inplace=True)

plt.scatter(df['Normalized_Change'], df['Label'])
plt.show()

# FEATURES = preprocessing.scale(np.array(df.drop(['label'], axis=1)))
# features = FEATURES[:-FORCAST_DAYS]
# features_after = FEATURES[-FORCAST_DAYS:]

# label = np.array(df['label'])
# label = label[:-FORCAST_DAYS]

# x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=TEST_SIZE)
# clf = LinearRegression(n_jobs=-1)
# clf.fit(x_train, y_train)
# accuracy = clf.score(x_test, y_test)
# print(f'Accuracy: {accuracy}.')

# forcast_set = clf.predict(features_after)
# print(forcast_set, accuracy, FORCAST_DAYS)