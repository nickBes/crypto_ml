import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

FORCAST_DAYS = 1
TEST_SIZE = 0.9
TEST_DAYS = 5 * FORCAST_DAYS

df = pd.read_csv('./bitcoin_clean.csv')

# Creating a label column for the prediction by moving the Close column 
# backwards by the desired amount of forcast days. In this way the feature columns will
# point to the close price in the future.

df['Label'] = df['Close'].shift(-FORCAST_DAYS)
df.dropna(inplace=True)

# We're creating a features array that doesn't include the time and 
# label and then we're seperating it into the rows that we want to train 
# and into the rows that we'll use for prediction.

FEATURES = preprocessing.scale(np.array(df.drop(['Label', 'Timestamp'], axis=1)))
features = FEATURES[:-TEST_DAYS]
features_after = FEATURES[-TEST_DAYS:]

# Creating a label array for training without the days that will be predicted.

label = np.array(df['Label'])
label = label[:-TEST_DAYS]

# The previous data is being prepared for training and testing.

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=TEST_SIZE)

# Creating a linear regression classifier, training and testing it. 

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(f'Accuracy: {accuracy}.')

# Using the classifier to predict the data and comparing it to the real data.
forcast_set = clf.predict(features_after)
end = df[-TEST_DAYS:]

# Getting from the user how much money he wants to trade, then we're checking for every day
# if the person should buy or sell by the prediction.

money = int(input('Money (in usd): '))
pm = money
print('')
btc = 0
si = 0
for index, row in end.iterrows():
    if money > 0:
        if row['Close'] < forcast_set[si]:
            print(f"Day {si + 1}, profit expected. Buying.")
            btc = money / row['Close']
            money = 0
    elif money == 0 and btc > 0:
        if row['Close'] < forcast_set[si]:
            print(f"Day {si + 1}, profit expected. Selling.")
            money = btc * row['Close']
            btc = 0
    si += 1

    if si == TEST_DAYS:
        last = row['Label']

# If there's some btc left sell on the next day and print the income

if btc > 0:
    money = last * btc
print("\nProfit: ", money - pm)