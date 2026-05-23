import yfinance as yf
import pandas as pd
import numpy as np
import sys
sys.path.append('C:/Users/ADMIN/.vscode/MachineLearningLibraries/Supervised')
from LinearRegression import LinearReg, train_test_split
import requests

# Download stock data
stock = yf.download("AAPL", start="2025-01-01", end="2026-05-22")
stock.columns = stock.columns.get_level_values(0)

# Create features
stock['Tomorrow'] = stock['Close'].shift(-1)
stock['Returns'] = stock['Close'].pct_change()
stock = stock.dropna()

# Fetch sentiment
api_key = 'I24B5B1ITX08B1OV'
url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={api_key}'
response = requests.get(url)
data = response.json()

scores = [article['overall_sentiment_score'] for article in data['feed']]
avg_sentiment = np.mean(scores)
print(f"Average sentiment: {avg_sentiment:.4f}")

# Add sentiment as feature
stock['Sentiment'] = avg_sentiment

# Define features and target
feature_cols = ['Volume', 'Returns', 'Sentiment']
x = stock[feature_cols].values
y = stock['Tomorrow'].values

print(f"x shape: {x.shape}")

# Normalize only Volume and Returns, not Sentiment
x_price = stock[['Volume', 'Returns']].values
x_price = (x_price - x_price.mean(axis=0)) / x_price.std(axis=0)
x_sentiment = stock[['Sentiment']].values
x = np.hstack([x_price, x_sentiment])

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train
model = LinearReg(0.01, 3000)
model.update(x_train, y_train)

# Evaluate
print(f"Train R2: {model.r2_score(x_train, y_train):.4f}")
print(f"Test R2: {model.r2_score(x_test, y_test):.4f}")

#PLOT THE PREDICTIONS
import matplotlib.pyplot as plt

x_all = np.hstack([
    (stock[['Volume', 'Returns']].values - stock[['Volume', 'Returns']].values.mean(axis=0)) / stock[['Volume', 'Returns']].values.std(axis=0),
    stock[['Sentiment']].values
])

predicted = model.predict(x_all)
actual = stock['Tomorrow'].values

plt.figure(figsize=(12, 5))
plt.plot(actual, label='Actual', color='blue')
plt.plot(predicted, label='Predicted', color='red')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Trading Days')
plt.ylabel('Price')
plt.legend()
plt.show()