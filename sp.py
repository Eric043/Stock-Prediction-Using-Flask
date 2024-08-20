from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

def get_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None, f"No data found for ticker {ticker}"
        return data, None
    except Exception as e:
        return None, str(e)

def train_and_predict(ticker, start_date, end_date):
    data, error = get_stock_data(ticker, start_date, end_date)
    
    if error:
        return None, None, None, error
    
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
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    last_value = scaled_data[-look_back:]  # Get the last value to predict tomorrow
    predicted_tomorrow_price = model.predict(last_value)
    predicted_tomorrow_price = scaler.inverse_transform(predicted_tomorrow_price.reshape(-1, 1))

    predicted_stock_price = model.predict(X)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))
    actual_stock_price = scaler.inverse_transform(y.reshape(-1, 1))
    
    return actual_stock_price, predicted_stock_price, predicted_tomorrow_price, None

def plot_results(actual_stock_price, predicted_stock_price, ticker, tomorrow_date):
    if actual_stock_price is None or predicted_stock_price is None:
        return None

    plt.figure(figsize=(14, 5))
    plt.plot(actual_stock_price, color='red', label='Today Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted Tomorrow Stock Price')
    plt.title(f'{ticker} Stock Price Prediction as of {tomorrow_date}')
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
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    # Calculate today's, yesterday's, and tomorrow's dates
    today_date = datetime.today().strftime('%Y-%m-%d')
    yesterday_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    tomorrow_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    actual_stock_price, predicted_stock_price, predicted_tomorrow_price, error = train_and_predict(ticker, start_date, end_date)
    
    if error:
        return jsonify({'error': error})
    
    img_str = plot_results(actual_stock_price, predicted_stock_price, ticker, tomorrow_date)
    
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
