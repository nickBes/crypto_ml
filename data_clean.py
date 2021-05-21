from datetime import datetime
import pandas as pd

sec_in_day = 86400

# Loading the csv to a dataframe, droping unecessary columns 
# (which were researched before) and rows that include nan.

df = pd.read_csv('./bitcoin.csv')
print(f'Initial length of the columns: {len(df)}.\n')
df.dropna(inplace=True)
df = df[['Timestamp', 'Close', 'Open', 'Volume_(BTC)']]

num = 0
to_drop = []
before = None
date0 = datetime.now()

# Filtering the dataframe to have timesteps of one day 
# and printing the progress every 25,000 rows.

for index, row in df.iterrows():
    num += 1
    delta = (datetime.now() - date0).total_seconds()
    if (num % 25000 == 0):
        print(f'Progress: {num}, Time: {delta}')

    if before is not None:
        if (row['Timestamp'] - before['Timestamp']) < sec_in_day:
            before['Volume_(BTC)'] += row['Volume_(BTC)']
            before['Close'] = row['Close']
            to_drop.append(index)
            continue
    before = row


df = df.drop(to_drop)
print(f'\nAfter clearing length: {len(df)}.\n')

# Getting ready the dataframe for saving it as csv: 
# creating a normalized change column and deleting the Open column.

df['Normalized_Change'] = (df['Open'] - df['Close'])/ df['Open']
df = df.drop(['Open'], axis=1)
print(f'Dataframe structure tail:\n\n{df.tail()}\n')
print(f'Dataframe structure head:\n\n{df.head()}\n')

df.to_csv('bitcoin_clean.csv')