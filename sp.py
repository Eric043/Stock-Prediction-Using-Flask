from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

def get_stock_data(ticker):
    try:
        # Download data from a year ago to the present
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None, f"No data found for ticker {ticker}"
        return data, end_date
    except Exception as e:
        return None, str(e)

def train_and_predict(ticker):
    data, end_date = get_stock_data(ticker)
    
    if data is None:
        return None, None, None, end_date  # end_date here represents the error message
    
    if data.shape[0] <= 1:
        return None, None, None, "Not enough data to proceed."
    
    data = data[['Close']]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    look_back = 1

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for Conv1D input
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(look_back, 1)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    last_value = scaled_data[-look_back:].reshape((1, look_back, 1))  # Reshape for Conv1D input
    predicted_tomorrow_price = model.predict(last_value)
    predicted_tomorrow_price = scaler.inverse_transform(predicted_tomorrow_price.reshape(-1, 1))

    predicted_stock_price = model.predict(X)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
    actual_stock_price = scaler.inverse_transform(y.reshape(-1, 1))
    
    return actual_stock_price, predicted_stock_price, predicted_tomorrow_price, end_date

def plot_results(actual_stock_price, predicted_stock_price, ticker, today_date):
    if actual_stock_price is None or predicted_stock_price is None:
        return None

    plt.figure(figsize=(14, 5))
    plt.plot(actual_stock_price, color='red', label='Today Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Tomorrow Stock Price')
    plt.title(f'{ticker} Stock Price Prediction as of {today_date}')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker')
    
    actual_stock_price, predicted_stock_price, predicted_tomorrow_price, end_date = train_and_predict(ticker)
    
    if end_date and "No data found" in end_date:
        return jsonify({'error': end_date})
    
    if actual_stock_price is None or predicted_stock_price is None:
        return jsonify({'error': 'Not enough data to proceed.'})
    
    # Calculate dynamic dates
    today_date = end_date
    yesterday_date = (datetime.strptime(today_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    tomorrow_date = (datetime.strptime(today_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    
    img_str = plot_results(actual_stock_price, predicted_stock_price, ticker, today_date)
    
    actual_recent_price = f"{actual_stock_price[-1][0]:.2f}" if actual_stock_price is not None and len(actual_stock_price) > 0 else "N/A"
    predicted_recent_price = f"{predicted_stock_price[-1][0]:.2f}" if predicted_stock_price is not None and len(predicted_stock_price) > 0 else "N/A"
    predicted_tomorrow_price = f"{predicted_tomorrow_price[0][0]:.2f}" if predicted_tomorrow_price is not None else "N/A"

    return jsonify({
        'image': img_str,
        'actual_recent_price': actual_recent_price,
        'predicted_recent_price': predicted_recent_price,
        'predicted_tomorrow_price': predicted_tomorrow_price,
        'today_date': today_date,
        'yesterday_date': yesterday_date,
        'tomorrow_date': tomorrow_date
    })

if __name__ == '__main__':
    app.run(debug=True)
