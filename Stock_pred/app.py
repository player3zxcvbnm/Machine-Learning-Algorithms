from flask import Flask, request
import sys
sys.path.append('C:/Users/ADMIN/.vscode/MachineLearningLibraries/Supervised')
from StockPredictor import get_prediction

app = Flask(__name__)

API_KEY = '6QTPD5U6HHVZOKQD'

@app.route('/')
def home():
    return '''
    <h1>Stock Predictor</h1>
    <form method="POST" action="/predict">
        <input type="text" name="ticker" placeholder="Enter ticker (e.g. AAPL)">
        <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    prediction, current, sentiment = get_prediction(ticker, API_KEY)
    return f'''
    <h1>{ticker} Prediction</h1>
    <p>Current Price: ${current:.2f}</p>
    <p>Predicted Tomorrow: ${prediction:.2f}</p>
    <p>Sentiment: {sentiment:.4f}</p>
    <a href="/">Back</a>
    '''

if __name__ == '__main__':
    app.run(debug=True)