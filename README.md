# Machine Learning Algorithms

## Stock Sentiment Predictor
Predicts next day stock price using linear regression built from scratch with NumPy, combining price data and news sentiment.

### Live Demo
https://linear-stock-predictor.onrender.com

Enter any stock ticker (AAPL, GOOGL, TSLA, MSFT) to get tomorrow's predicted price and sentiment score.

### How it works
- Fetches real price data via yfinance
- Fetches news sentiment via Alpha Vantage API
- Trains linear regression from scratch (no sklearn)
- R2 = 0.12 on test set

### How to run locally
1. Install dependencies: `pip install flask yfinance numpy pandas requests`
2. Add your Alpha Vantage API key to Render environment variables
3. Run: `python Stock_pred/app.py`

## ML Algorithm Library
Supervised learning algorithms built from scratch:
- Linear Regression (gradient descent, normalization, train/test split)
- Logistic Regression (sigmoid, cross entropy loss, accuracy scoring)
