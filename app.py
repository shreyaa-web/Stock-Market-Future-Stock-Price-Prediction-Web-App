from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA  # Updated import
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
import requests
import os
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt', quiet=True)
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key_for_dev')  # Added secret key

# Create data directory if it doesn't exist
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Alpha Vantage API key from environment variables
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'YF43BPYIAPDH2YRT')  # Default for backward compatibility

# Helper function to validate stock ticker
def is_valid_ticker(ticker):
    """Validate if the ticker symbol is valid."""
    if not ticker or not isinstance(ticker, str) or len(ticker) > 10:
        return False
    # Basic validation - alphanumeric and dot
    return bool(re.match(r'^[A-Za-z0-9.]+$', ticker))

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route("/")
def logstylepage():
    return render_template('signlog.html')

@app.route("/prediction", methods=['POST'])
def prediction():
    nm = request.form.get('nm', '')
    
    # Validate ticker symbol
    if not is_valid_ticker(nm):
        flash('Invalid ticker symbol. Please enter a valid stock symbol.')
        return render_template('signlog.html', not_found=True)
        
    quote = nm

    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        """Fetch historical stock data and save to CSV."""
        try:
            ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=quote, outputsize='full')
        
            # Format df - Last 2 yrs rows => 502, in ascending order => ::-1
            data = data.head(503).iloc[::-1]
            data = data.reset_index()
            
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Volume'] = data['5. volume']
            
            # Save to data directory
            csv_path = data_dir / f"{quote}.csv"
            df.to_csv(csv_path, index=False)
            return df
        except Exception as e:
            print(f"Error fetching data for {quote}: {str(e)}")
            return None

    #******************** ARIMA SECTION ********************
    def ARIMA_ALGO(df):
        """Perform ARIMA prediction."""
        try:
            uniqueVals = df["Code"].unique()  
            df = df.set_index("Code")
            
            # For daily basis
            def parser(x):
                # Handle both string and Timestamp objects
                if isinstance(x, str):
                    return datetime.strptime(x, '%Y-%m-%d')
                else:
                    # For pandas Timestamp objects, just return the datetime
                    return x.to_pydatetime()
                
            def arima_model(train, test):
                history = [x for x in train]
                predictions = list()
                for t in range(len(test)):
                    model = ARIMA(history, order=(6, 1, 0))
                    model_fit = model.fit()  # Updated fit method
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = test[t]
                    history.append(obs)
                return predictions
                
            for company in uniqueVals[:10]:
                data = (df.loc[company, :]).reset_index()
                data['Price'] = data['Close']
                Quantity_date = data[['Price', 'Date']]
                
                # Convert to datetime index safely
                Quantity_date.index = pd.to_datetime(Quantity_date['Date'])
                Quantity_date['Price'] = Quantity_date['Price'].astype(float)
                Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
                Quantity_date = Quantity_date.drop(['Date'], axis=1)
                
                # Use matplotlib's non-interactive backend to avoid Tkinter issues
                import matplotlib
                matplotlib.use('Agg')  # Set non-interactive backend
                
                fig = plt.figure(figsize=(7.2, 4.8), dpi=75)
                plt.plot(Quantity_date)
                plt.savefig('static/Trends.png')
                plt.close(fig)
                
                quantity = Quantity_date.values
                size = int(len(quantity) * 0.80)
                train, test = quantity[0:size], quantity[size:len(quantity)]
                
                # Fit in model
                predictions = arima_model(train, test)
                
                # Plot graph
                fig = plt.figure(figsize=(7.2, 4.8), dpi=75)
                plt.plot(test, label='Actual Price')
                plt.plot(predictions, label='Predicted Price')
                plt.legend(loc=4)
                plt.savefig('static/ARIMA.png')
                plt.close(fig)
                
                arima_pred = predictions[-2]
                print(f"Tomorrow's {quote} Closing Price Prediction by ARIMA: {arima_pred}")
                
                # RMSE calculation
                error_arima = math.sqrt(mean_squared_error(test, predictions))
                print(f"ARIMA RMSE: {error_arima}")
                
                return arima_pred, error_arima
                
        except Exception as e:
            print(f"Error in ARIMA prediction: {str(e)}")
            return 0, 0

    #************* LSTM SECTION **********************
    def LSTM_ALGO(df):
        """Perform LSTM prediction."""
        try:
            # Split data into training set and test set
            dataset_train = df.iloc[0:int(0.8 * len(df)), :]
            dataset_test = df.iloc[int(0.8 * len(df)):, :]
            
            # NOTE: To predict stock prices of next N days, store previous N days in memory while training
            # Here N=7
            training_set = df.iloc[:, 4:5].values  # 1:2, to store as numpy array else Series obj will be stored
            
            # Feature Scaling
            from sklearn.preprocessing import MinMaxScaler
            sc = MinMaxScaler(feature_range=(0, 1))  # Scaled values between 0,1
            training_set_scaled = sc.fit_transform(training_set)
            
            # Creating data structure with 7 timesteps and 1 output
            X_train = []  # Memory with 7 days from day i
            y_train = []  # Day i
            for i in range(7, len(training_set_scaled)):
                X_train.append(training_set_scaled[i-7:i, 0])
                y_train.append(training_set_scaled[i, 0])
                
            # Convert list to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_forecast = np.array(X_train[-1, 1:])
            X_forecast = np.append(X_forecast, y_train[-1])
            
            # Reshaping: Adding 3rd dimension
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
            
            # Set TensorFlow to suppress various warnings
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            
            # Building RNN with TensorFlow 2.x style
            regressor = Sequential()
            
            # Add first LSTM layer
            regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            regressor.add(Dropout(0.1))
            
            # Add 2nd LSTM layer
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.1))
            
            # Add 3rd LSTM layer
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.1))
            
            # Add 4th LSTM layer
            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.1))
            
            # Add o/p layer
            regressor.add(Dense(units=1))
            
            # Compile
            regressor.compile(optimizer='adam', loss='mean_squared_error')
            
            # Suppress TensorFlow's verbose output
            import logging
            tf_logger = logging.getLogger('tensorflow')
            tf_logger.setLevel(logging.ERROR)
            
            # Training with simplified output
            print(f"Training LSTM model for {quote}...")
            regressor.fit(
                X_train, y_train,
                epochs=25,
                batch_size=32,
                verbose=0  # Silent mode
            )
            print(f"LSTM model training complete.")
            
            # Testing
            real_stock_price = dataset_test.iloc[:, 4:5].values
            
            # To predict, we need stock prices of 7 days before the test set
            dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0) 
            testing_set = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values
            testing_set = testing_set.reshape(-1, 1)
            
            # Feature scaling
            testing_set = sc.transform(testing_set)
            
            # Create data structure
            X_test = []
            for i in range(7, len(testing_set)):
                X_test.append(testing_set[i-7:i, 0])
                
            # Convert list to numpy arrays
            X_test = np.array(X_test)
            
            # Reshaping: Adding 3rd dimension
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Testing Prediction
            predicted_stock_price = regressor.predict(X_test, verbose=0)
            
            # Getting original prices back from scaled values
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            
            fig = plt.figure(figsize=(7.2, 4.8), dpi=75)
            plt.plot(real_stock_price, label='Actual Price')  
            plt.plot(predicted_stock_price, label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig('static/LSTM.png')
            plt.close(fig)
            
            error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
            
            # Forecasting Prediction
            forecasted_stock_price = regressor.predict(X_forecast, verbose=0)
            
            # Getting original prices back from scaled values
            forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
            
            lstm_pred = forecasted_stock_price[0, 0]
            print(f"Tomorrow's {quote} Closing Price Prediction by LSTM: {lstm_pred}")
            print(f"LSTM RMSE: {error_lstm}")
            
            return lstm_pred, error_lstm
        except Exception as e:
            print(f"Error in LSTM prediction: {str(e)}")
            return 0, 0

    #***************** LINEAR REGRESSION SECTION ******************       
    def LIN_REG_ALGO(df):
        """Perform Linear Regression prediction."""
        try:
            # Make a copy to avoid modifying the original dataframe
            df_copy = df.copy()
            
            # No of days to be forecasted in future
            forecast_out = int(7)
            # Price after n days
            df_copy['Close after n days'] = df_copy['Close'].shift(-forecast_out)
            # New df with only relevant data
            df_new = df_copy[['Close', 'Close after n days']]
            
            # Handle any NaN values
            df_new = df_new.dropna()
            
            # If there's not enough data after dropping NaNs, return default values
            if len(df_new) < 10:
                print("Not enough data for Linear Regression prediction after removing NaN values")
                return df_copy, 0, np.array([0]*7).reshape(-1,1), 0, 0

            # Structure data for train, test & forecast
            # Labels of known data, discard last rows equal to forecast_out
            y = np.array(df_new.iloc[:-forecast_out, -1])
            y = np.reshape(y, (-1, 1))
            # All cols of known data except labels, discard last rows equal to forecast_out
            X = np.array(df_new.iloc[:-forecast_out, 0:-1])
            # Unknown, X to be forecasted
            X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])
            
            # Make sure we have enough data for training and testing
            train_size = int(0.8 * len(X))
            if train_size < 1 or (len(X) - train_size) < 1:
                print("Not enough data points for training/testing split")
                return df_copy, 0, np.array([0]*7).reshape(-1,1), 0, 0
                
            # Training, testing to plot graphs, check accuracy
            X_train = X[0:train_size, :]
            X_test = X[train_size:, :]
            y_train = y[0:train_size, :]
            y_test = y[train_size:, :]
            
            # Feature Scaling===Normalization
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            X_to_be_forecasted = sc.transform(X_to_be_forecasted)
            
            # Training
            clf = LinearRegression(n_jobs=-1)
            clf.fit(X_train, y_train)
            
            # Testing
            y_test_pred = clf.predict(X_test)
            y_test_pred = y_test_pred * (1.04)
            
            fig = plt.figure(figsize=(7.2, 4.8), dpi=75)
            plt.plot(y_test, label='Actual Price')
            plt.plot(y_test_pred, label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig('static/LR.png')
            plt.close(fig)
            
            error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Forecasting
            forecast_set = clf.predict(X_to_be_forecasted)
            forecast_set = forecast_set * (1.04)
            mean = forecast_set.mean()
            lr_pred = forecast_set[0, 0]
            
            print(f"Tomorrow's {quote} Closing Price Prediction by Linear Regression: {lr_pred}")
            print(f"Linear Regression RMSE: {error_lr}")
            
            return df_copy, lr_pred, forecast_set, mean, error_lr
        except Exception as e:
            print(f"Error in Linear Regression prediction: {str(e)}")
            return df, 0, np.array([0]*7).reshape(-1,1), 0, 0
    
    def sentanal(tic):
        """Perform sentiment analysis on news."""
        try:
            finviz_url = 'https://finviz.com/quote.ashx?t='
            tickers = [tic]

            news_tables = {}
            for ticker in tickers:
                url = finviz_url + ticker
                req = Request(url=url, headers={'user-agent': 'my-app'})
                response = urlopen(req)

                html = BeautifulSoup(response, features='html.parser')
                news_table = html.find(id='news-table')
                news_tables[ticker] = news_table
                
            parsed_data = []
            
            for ticker, news_table in news_tables.items():
                if news_table is None:
                    continue
                    
                for row in news_table.findAll('tr'):
                    if row.a is None:
                        continue
                    title = row.a.text
                    date_data = row.td.text.split(' ')
                    
                    if len(date_data) == 1:
                        time = date_data[0]
                    else:
                        date = date_data[0]
                        time = date_data[1]
                    
                    parsed_data.append([ticker, date, time, title])

            if not parsed_data:
                print("No news data found for", tic)
                return 0, parsed_data, "No News", 0, 0, 0

            df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
            try:
                df['date'] = pd.to_datetime(df.date).dt.date
            except:
                # Handle date format issues
                print("Using today's date for all news items")
                df['date'] = datetime.today().date()
                
            # Ensure NLTK resources are available
            try:
                vader = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"Error initializing VADER: {str(e)}")
                # Try downloading again
                nltk.download('vader_lexicon')
                vader = SentimentIntensityAnalyzer()
                
            pos = 0
            neg = 0
            neutral = 0
            compound_scores = []
            
            # Calculate sentiment for each title
            for title in df['title']:
                try:
                    score = vader.polarity_scores(title)['compound']
                    compound_scores.append(score)
                    if score > 0:
                        pos += 1
                    elif score < 0:
                        neg += 1
                    else:
                        neutral += 1
                except Exception as e:
                    print(f"Error analyzing title: {str(e)}")
                    compound_scores.append(0)
                    neutral += 1
            
            df['compound'] = compound_scores
            print(f"Positive News: {pos}, Negative News: {neg}, Neutral News: {neutral}")
            
            # Calculate global polarity directly
            global_polarity = df['compound'].mean() if len(df) > 0 else 0
            print(f"Global polarity for {tic}: {global_polarity}")
            
            # Create a simple bar chart showing sentiment distribution
            plt.figure(figsize=(9.2, 6.8))
            plt.bar(['Positive', 'Negative', 'Neutral'], [pos, neg, neutral])
            plt.title(f'Sentiment Analysis for {tic}')
            plt.savefig('static/SA.png', bbox_inches='tight', dpi=80)
            plt.close()
            
            overall_poll = ""
            if global_polarity > 0:
                print("News Polarity: Overall Positive")
                overall_poll = "Overall Positive"
            else:
                print("News Polarity: Overall Negative")
                overall_poll = "Overall Negative"
                
            news_list = parsed_data
            return global_polarity, news_list, overall_poll, pos, neg, neutral
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            # Return default values
            return 0, [], "Error - No Sentiment Data Available", 0, 0, 0

    def recommending(df, global_polarity, today_stock, mean):
        """Generate trading recommendation based on predictions and sentiment."""
        try:
            if today_stock.iloc[-1]['Close'] < mean:
                if global_polarity > 0:
                    idea = "RISE"
                    decision = "BUY"
                    print(f"According to the ML Predictions and Sentiment Analysis of News, a {idea} in {quote} stock is expected => {decision}")
                elif global_polarity <= 0:
                    idea = "FALL"
                    decision = "SELL"
                    print(f"According to the ML Predictions and Sentiment Analysis of News, a {idea} in {quote} stock is expected => {decision}")
            else:
                idea = "FALL"
                decision = "SELL"
                print(f"According to the ML Predictions and Sentiment Analysis of News, a {idea} in {quote} stock is expected => {decision}")
            return idea, decision
        except Exception as e:
            print(f"Error in recommendation: {str(e)}")
            return "UNKNOWN", "HOLD"

    try:
        # Get historical data
        df = get_historical(quote)
        if df is None:
            flash(f"Could not retrieve data for {quote}. Please try another ticker.")
            return render_template('signlog.html', not_found=True)
        
        # Preprocess data
        print(f"Today's {quote} Stock Data: ")
        today_stock = df.iloc[-1:]
        print(today_stock)
        
        df = df.dropna()
        code_list = []
        for i in range(0, len(df)):
            code_list.append(quote)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df = df2

        # Run predictions
        arima_pred, error_arima = ARIMA_ALGO(df)
        lstm_pred, error_lstm = LSTM_ALGO(df)
        df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df)
        
        # Perform sentiment analysis
        polarity, tw_list, tw_pol, pos, neg, neutral = sentanal(quote)
        
        # Generate recommendation
        idea, decision = recommending(df, polarity, today_stock, mean)
        
        print("Forecasted Prices for Next 7 days:")
        print(forecast_set)
        
        # Round values for display
        today_stock = today_stock.round(2)
        
        return render_template('index.html', 
                              quote=quote,
                              arima_pred=round(arima_pred, 2),
                              lstm_pred=round(lstm_pred, 2),
                              lr_pred=round(lr_pred, 2),
                              open_s=today_stock['Open'].to_string(index=False),
                              close_s=today_stock['Close'].to_string(index=False),
                              tw_list=tw_list,
                              tw_pol=tw_pol,
                              idea=idea,
                              decision=decision,
                              high_s=today_stock['High'].to_string(index=False),
                              low_s=today_stock['Low'].to_string(index=False),
                              vol=today_stock['Volume'].to_string(index=False),
                              forecast_set=forecast_set,
                              error_lr=round(error_lr, 2),
                              error_lstm=round(error_lstm, 2),
                              error_arima=round(error_arima, 2))
                              
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return render_template('signlog.html', not_found=True)

if __name__ == '__main__':
   app.run(debug=False)