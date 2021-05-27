
# crypto_ml

Welcome to the **crypto_ml** repository. This repository stores the source code of the final AI project for **neta@**. The goal of this project is to develop an AI that predicts the future prices of **BTC** and simulate the income or loss from buying/selling bitcoin by the AI's predictions.

  

# Getting started

In order to use the code you need to install all of the required files and dependencies using the terminal:

- First, clone the **GitHub** repository:

`git clone https://github.com/nickBes/crypto_ml.git`

- Then, enter the new folder:

`cd crypto_ml`

- In the folder create a python virtual environment:

`python -m venv env`

- Activate the python environment:

- For **Windows** -

`env\Scripts\activate`

- For **Linux**/**MacOs** -

`source ./env/bin/activate`

- Install the required packages:

`pip install requirements.txt`

  

Now you're ready to use the code.

# Files

## `main.py`

Here we're training a linear regression classifier using our data in the `bitcoin_clean.csv` file and then we're simulating the profit / loss in the next couple of days.
## `data_clean.py`

After downloading the full **CSV** file of the bitcoin data from this [link](https://www.kaggle.com/mczielinski/bitcoin-historical-data), running the `data_clean.py` file would clean the csv from unnecessary data and saves it to a new csv file named `bitcoin_clean.csv`.

Keep in mind that after downloading the csv you should rename it into `bitcoin.csv`.