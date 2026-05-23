# Stock Sentiment Predictor

Predicts next day AAPL price using volume, daily returns, and news sentiment.

## How it works
- Fetches price data via yfinance
- Fetches news sentiment via Alpha Vantage API
- Trains linear regression from scratch (no sklearn)

## Results
- Without sentiment: R2 = 0.02
- With sentiment: R2 = 0.12

## How to run
pip install yfinance requests numpy matplotlib
python StockPredictor.py

## Limitations
Linear regression misses price momentum. Future: LSTM
