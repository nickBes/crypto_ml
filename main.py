import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

FORCAST_DAYS = 10
TEST_SIZE = 0.2

df = pd.read_csv('./coin_Bitcoin.csv')
print(f'Initial length of the columns: {len(df)}.\n')

# The volume column is empty is we're recalculating the volume.
# We also calculate the normalized change of the cryptocurrency value

df['Volume'] = df['Marketcap'] / df['Close']
df['NormChange'] = (df['Open'] - df['Close']) / df['Open']
df = df[['Close', 'NormChange', 'Volume']]

# Creating a label column for the prediction by moving the Close column 
# backwards by the desired amount of forcast days.

df['label'] = df['Close'].shift(-FORCAST_DAYS)
print(f'Dataframe structure:\n\n{df.tail()}\n')

FEATURES = preprocessing.scale(np.array(df.drop(['label'], axis=1)))
features = FEATURES[:-FORCAST_DAYS]
features_after = FEATURES[-FORCAST_DAYS:]

label = np.array(df['label'])
label = label[:-FORCAST_DAYS]

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=TEST_SIZE)
clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(f'Accuracy: {accuracy}.')

forcast_set = clf.predict(features_after)
print(forcast_set, accuracy, FORCAST_DAYS)

plt.plot(df['Close'])
plt.show()