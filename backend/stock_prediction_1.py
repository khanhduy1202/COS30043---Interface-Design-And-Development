# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow as tf
import yfinance as yf
import mplfinance as mf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, InputLayer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def shuffle_in_unison(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def load_data(datasource, ticker, start_date, end_date, test_size=0.2, scale= True, random_split=False,
              save_data=True, load_saved_data=False, save_scalers=False, prediction_days=60, lookup_step=10, price_value='Close', shuffle=True):
    # Load data from source or use local data
    if load_saved_data:
        df = pd.read_csv('local_data.csv')
    else:
        if isinstance(ticker, str):
            data = yf.download(ticker, start=start_date, end=end_date)
            df = data.reset_index()
        elif isinstance(ticker, pd.DataFrame):
            df = ticker
        else:
            raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
        #save data to local file
        if save_data:
            df.to_csv('local_data.csv', index=False)

    print(df.columns)
    # Define feature columns
    feature_columns = ['Adj Close', 'Volume', 'Open', 'Close', 'High', 'Low']

    # Add date col
    if "Date" not in df.columns:
        df["Date"] = df.set_index

    # Check col existence
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # Drop NaNs
    df.dropna(inplace=True)
    og_df = df.copy() #original data

    # add the target column (label) by shifting by `lookup_step`
    df['Future'] = df['Adj Close'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    n_steps=50
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["Date"]].values, df['Future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)

    # Scale features
    if scale:
        column_scalers = {}
        for column in feature_columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
            scaled_data = scaler.fit_transform(df[feature_columns].values.reshape(-1, 1))
            scaled_data = scaled_data[:,0]
            column_scalers[column] = scaler

    # Prepare training data
    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x])
        y_train.append(scaled_data[x])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) #reshape to 3D array

    # Split data randomly
    if random_split:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, shuffle=shuffle)
    else:
        split_index = int(len(df) * (1-test_size))
        x_train = x_train[:split_index]
        x_test = x_train[split_index:]
        y_train = y_train[:split_index]
        y_test = y_train[split_index:]
        train_data = data[:split_index]
        test_data = data[split_index:]
        # Apply the corresponding scaler to each feature in the test data
        if shuffle:
            shuffle_in_unison(x_train, y_train)
            shuffle_in_unison(x_test, y_test)
    
    # Return the result
    result = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'column_scalers': column_scalers,
        'df': df,
        'og_df': og_df,
        'last_sequence': last_sequence,
        'train_data': train_data,
        'test_data': test_data
    }

    return result

#----------------------------------------------------------------------------
def multivariate_multistep_predict(model, input_sequence, k, scaler):
    predicted_prices = []
    current_sequence = input_sequence  # Initialize with the input sequence

    for _ in range(k):
        # Predict the following days closing price
        prediction = model.predict(current_sequence)
        prediction = scaler.inverse_transform(prediction)
        predicted_prices.append(prediction)
        # create a new sequence to maintain its shape
        new_sequence = np.append(current_sequence[0, 1:], prediction)
        new_sequence = new_sequence.reshape((1, new_sequence.shape[0], 1))
        current_sequence = new_sequence

    predicted_prices = np.array(predicted_prices)

    return predicted_prices

#------------------------------------------------------------------------------

def plot_candlestick_chart(data, title="Candlestick Chart", n_days=1):
    
    # Resample the data to represent 'n' days in each candlestick
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    resampled_data = data.resample(f'{n_days}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    })

    # Create the candlestick chart
    mf.plot(resampled_data, type='candle', title=title, ylabel='Price', style='yahoo')
    mf.show()

#------------------------------------------------------------------------------

def plot_boxplot(data, title="Boxplot Chart", num_windows=20):
    '''
    Create a boxplot chart from the given data.

    Parameters:
    - boxplot_data: List of arrays
        Data for each boxplot.
    - labels: List of str
        Labels for each boxplot.
    '''
    boxplot_data = []
    for i in range(num_windows):
        data = np.random.rand(50)
        boxplot_data.append(data)
    plt.figure(figsize=(10, 6))
    plt.boxplot(boxplot_data, labels=labels)
    plt.title("Boxplot Chart")
    plt.xlabel("Window")
    plt.ylabel("Closing Price")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Generate mock candlestick chart data
def generate_candlestick_chart_data(start_date, num_days=30):
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    open_prices = np.random.rand(num_days)
    high_prices = open_prices + np.random.rand(num_days)
    low_prices = open_prices - np.random.rand(num_days)
    close_prices = open_prices + np.random.rand(num_days)

    data = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices
    })

    return data

# Generate mock boxplot chart data
def generate_boxplot_chart_data(num_windows=20, data_points_per_window=50):
    chart_data = []
    for _ in range(num_windows):
        data = np.random.rand(data_points_per_window)
        chart_data.append(data.tolist())

    return chart_data

#------------------------------------------------------------------------------



def create_model(units=50, cell=LSTM, n_layers=2, dropout=0.2, loss="mean_squared_error", optimizer="adam", bidirectional=False, x_train=[]):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    #model.save_weights("model_file.h5") #Save the model file
    return model

#--------------------------------------------------------------------------------------------------
#Plotting charts

# Actual and Predicted data chart
'''
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(f'Shape of x_test: {x_test.shape}')
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()
'''

#Candlestick Chart
#plot_candlestick_chart(chart_data)

# Boxplot Chart

'''num_windows=18
labels = [f"Window {i+1}" for i in range(num_windows)]
plot_boxplot(chart_data, labels, num_windows)
'''