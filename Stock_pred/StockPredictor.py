import yfinance as yf
import pandas as pd
import numpy as np
import sys
sys.path.append('C:/Users/ADMIN/.vscode/MachineLearningLibraries/Supervised')
from LinearRegression import LinearReg, train_test_split
import requests

def get_prediction(ticker, api_key):
    # Download stock data
    stock = yf.download(ticker, start="2025-01-01", end="2026-05-22")
    stock.columns = stock.columns.get_level_values(0)
    stock['Tomorrow'] = stock['Close'].shift(-1)
    stock['Returns'] = stock['Close'].pct_change()
    stock = stock.dropna()

    # Fetch sentiment
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    scores = [a['overall_sentiment_score'] for a in data['feed']]
    avg_sentiment = np.mean(scores)

    # Add sentiment as feature
    stock['Sentiment'] = avg_sentiment

    # Normalize Volume and Returns only
    x_price = stock[['Volume', 'Returns']].values
    x_price = (x_price - x_price.mean(axis=0)) / x_price.std(axis=0)
    x = np.hstack([x_price, stock[['Sentiment']].values])
    y = stock['Tomorrow'].values

    # Split and train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearReg(0.01, 3000)
    model.update(x_train, y_train)

    # Predict latest
    latest = x[-1].reshape(1, -1)
    prediction = model.predict(latest)[0]
    current = stock['Close'].iloc[-1]

    return prediction, current, avg_sentiment