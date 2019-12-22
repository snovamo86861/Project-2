#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import dump, load
from fuzzywuzzy import process
import requests as requests
import warnings
warnings.filterwarnings('ignore')

from botocore.vendored import requests

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


### Helper functions ###


# Load model for results   ####go back to intent handler 
def get_model():
    model = load('project2_random_forest_model.joblib')
    return model 

# Converts a non-numeric value to float
def parse_float(n):
    try:
        return float(n)
    except ValueError:
        return float("nan")

    
# Get list of stocks
def getCompany(text):
    
    url="https://api.iextrading.com/1.0/ref-data/symbols"
    r = requests.post(url)
    stockList = r.json() 
    
    return process.extractOne(text, stockList)[0]


# Create final dataframe
def create_df(text, data):
    
    company = getCompany(company_request)
    symbol = company['symbol']
    company_name = company['name']
    data = yf.download(symbol, parse_dates=True, infer_datetime_format=True)
    data['Daily Return'] = data['Close'].dropna().pct_change()
    
    return data

def last_close_price(data, data_last_close):
    data_last_close = data[-1:]['Close']
    return(data_last_close)

# Construct exponential_moving_average function
def exponential_moving_average(data):
    
    # Set short and long windows
    short_window = 50
    long_window = 100

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    data['fast_close'] = data['Close'].ewm(halflife=short_window).mean()
    data['slow_close'] = data['Close'].ewm(halflife=long_window).mean()

    # Construct a crossover trading signal
    data['crossover_long'] = np.where(data['fast_close'] > data['slow_close'], 1.0, 0.0)
    data['crossover_short'] = np.where(data['fast_close'] < data['slow_close'], -1.0, 0.0)
    data['crossover_signal'] = data['crossover_long'] + data['crossover_short']

    short_vol_window = 50
    long_vol_window = 100

    # Construct a `Fast` and `Slow` Exponential Moving Average from short and long windows, respectively
    data['fast_vol'] = data['Daily Return'].ewm(halflife=short_vol_window).std()
    data['slow_vol'] = data['Daily Return'].ewm(halflife=long_vol_window).std()

    # Construct a crossover trading signal
    data['vol_trend_long'] = np.where(data['fast_vol'] < data['slow_vol'], 1.0, 0.0)
    data['vol_trend_short'] = np.where(data['fast_vol'] > data['slow_vol'], -1.0, 0.0) 
    data['vol_trend_signal'] = data['vol_trend_long'] + data['vol_trend_short']

    # Set bollinger band window
    bollinger_window = 20

    # Calculate rolling mean and standard deviation
    data['bollinger_mid_band'] = data['Close'].rolling(window=bollinger_window).mean()
    data['bollinger_std'] = data['Close'].rolling(window=20).std()

    # Calculate upper and lowers bands of bollinger band
    data['bollinger_upper_band']  = data['bollinger_mid_band'] + (data['bollinger_std'] * 1)
    data['bollinger_lower_band']  = data['bollinger_mid_band'] - (data['bollinger_std'] * 1)

    # Calculate bollinger band trading signal
    data['bollinger_long'] = np.where(data['Close'] < data['bollinger_lower_band'], 1.0, 0.0)
    data['bollinger_short'] = np.where(data['Close'] > data['bollinger_upper_band'], -1.0, 0.0)
    data['bollinger_signal'] = data['bollinger_long'] + data['bollinger_short']

    # Set the short window and long windows
    rolling_short_window = 50
    rolling_long_window = 100

    # Generate the short and long moving averages (50 and 100 days, respectively)
    data["SMA50"] = data["Close"].rolling(window=short_window).mean()
    data["SMA100"] = data["Close"].rolling(window=long_window).mean()

    # Initialize the new `Signal` column
    data["SMA_Signal"] = 0.0

    # Generate the trading signal 0 or 1,
    # where 0 is when the SMA50 is under the SMA100, and
    # where 1 is when the SMA50 is higher (or crosses over) the SMA100
    data["SMA_Signal"][short_window:] = np.where(
        data["SMA50"][short_window:] < data["SMA100"][short_window:], 1.0, 0.0
    )
    return data

# Set x variable list of features
def signals(signals):
    x_var_list = ['crossover_signal', 'vol_trend_signal', 'bollinger_signal', 'SMA_Signal']
    data[x_var_list] = data[x_var_list].shift(1)
    data[x_var_list].tail()

# Signal_data and replace positive/negative infinity values
def signal_for_data(data, signals):
         
    data.dropna(subset=x_var_list, inplace=True)
    data.dropna(subset=['Daily Return'], inplace=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    
    return(data, signals)

         
def positive_return():
    data['Positive Return'] = np.where(data['Daily Return'] > 0, 1.0, 0.0)

         
         
    """
    Figure how to connect user input date here
    """
         
# Construct dataset for predictions        
def model_predictions(data, model_predictions, start_date, end_date):
    start_date = data.index.min().strftime(format= '%Y-%m-%d')
    end_date = data.index.max().strftime(format= '%Y-%m-%d')

    x_data = data[x_var_list][start_date:end_date]
    predictions = model.predict(x_data)
    
    # Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
    data["Predicted Value"] = predictions
         
    return predictions, data
         
# Create a signals_df

def signals_df(data, signals_df, x, negative_check):
         
    # Grab just the `date` and `close` from the IEX dataset
    signals_df = data.loc[:, ['Entry Exit', 'Close']].copy()

    signals_df["Entry Exit"][0] = 0
    signals_df['Position'] = 0
    signals_df['Entry/Exit Position'] = 0
    signals_df['Portfolio Holdings'] = 0
    signals_df['Portfolio Cash'] = 0
    signals_df['Portfolio Total'] = 0
    signals_df['Portfolio Daily Returns'] = 0
    signals_df['Portfolio Cumulative Returns'] = 0

    signals_df["Entry Exit"][0] = 0

    negative_check = signals_df.loc[signals_df['Entry Exit'] != 0].reset_index()
    negative_check[:5]

    if negative_check.iloc[0]['Entry Exit'] < 0:
        x = negative_check.iloc[0]['Date'].strftime(format= '%Y-%m-%d')
        signals_df['Entry Exit'][x] = 0
    
    return signals_df, x, signals_df, negative_check

         
# Test the model for initial investment

def initial_investment(signals_df, portfolio_total_return, initial_capital, share_size):
         
    # Set initial capital
    initial_capital = float(investment_amount)
    signals_df['Portfolio Cash'][0] = initial_capital 
    share_size = int(shares_wanted)


    # Take a 500 share position where the dual moving average crossover is 1 (SMA50 is greater than SMA100)
    signals_df['Position'] = share_size * signals_df['Entry Exit']

    # Find the points in time where a 500 share position is bought or sold
    signals_df['Entry/Exit Position'] = signals_df['Position'].cumsum()

    signals_df['Total Entry/Exit Position'] = signals_df['Entry/Exit Position'].diff()

    # Multiply share price by entry/exit positions and get the cumulatively sum
    signals_df['Portfolio Holdings'] = signals_df['Close'] * signals_df['Total Entry/Exit Position'].cumsum()

    # Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
    signals_df['Portfolio Cash'] = initial_capital - (signals_df['Close'] * signals_df['Total Entry/Exit Position']).cumsum()

    # Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
    signals_df['Portfolio Total'] = signals_df['Portfolio Cash'] + signals_df['Portfolio Holdings']
    portfolio_total_return = signals_df['Portfolio Total']

    # Calculate the portfolio daily returns
    signals_df['Portfolio Daily Returns'] = signals_df['Portfolio Total'].pct_change()

    # Calculate the cumulative returns
    signals_df['Portfolio Cumulative Returns'] = (1 + signals_df['Portfolio Daily Returns']).cumprod() - 1
    portfolio_cum_return = signals_df['Portfolio Cumulative Returns']  
       
    return portfolio_total_return

def without_algo (signal_df, portfolio_total_return, initial_capital):
    
    # Return without algorithm
    signals_df['Without Algorithm'] = (int(initial_capital/signals_df['Close'][0])) * signals_df['Close']
    portfolio_without_algo = signals_df['Without Algorithm']
         
    return portfolio_without_algo
    


# In[ ]:


### Dialog Actions Helper Functions ###

def get_slots(intent_request):
    
    """
    Fetch all the slots and their values from the current intent.
    """
    return intent_request["currentIntent"]["slots"]


def elicit_slot(session_attributes, intent_name, slots, slot_to_elicit, message):
    """
    Defines an elicit slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "ElicitSlot",
            "intentName": intent_name,
            "slots": slots,
            "slotToElicit": slot_to_elicit,
            "message": message,
        },
    }


def delegate(session_attributes, slots):
    """
    Defines a delegate slot type response.
    """

    return {
        "sessionAttributes": session_attributes,
        "dialogAction": {"type": "Delegate", "slots": slots},
    }


def close(session_attributes, fulfillment_state, message):
    """
    Defines a close slot type response.
    """

    response = {
        "sessionAttributes": session_attributes,
        "dialogAction": {
            "type": "Close",
            "fulfillmentState": fulfillment_state,
            "message": message,
        },
    }

    return response


# In[ ]:


def build_validation_result(is_valid, violated_slot, message_content):
    """
    Defines an internal validation message structured as a python dictionary.
    """
    if message_content is None:
        return {"isValid": is_valid, "violatedSlot": violated_slot}

    return {
        "isValid": is_valid,
        "violatedSlot": violated_slot,
        "message": {"contentType": "PlainText", "content": message_content},
    }


def validate_data(investment_amount, shares_wanted, start_date, intent_request):
    
    """
    Validates the data provided by the user.
    """
    
    # Validate the investment amount, it should be > 0
    if investment_amount is not None:
        investment_amount = parse_float(
            investment_amount
        )  # Since parameters are strings it's important to cast values
        if investment_amount <= 0:
            return build_validation_result(
                False,
                "investment_amount",
                "The amount to invest should be greater than zero, "
                "please provide a correct amount in USD to invest.",
            )
        
    # Validate the number of shares, it should be > 0
    if shares_wanted is not None:
        shares_wanted = parse_float(
            shares_wanted
        )  # Since parameters are strings it's important to cast values
        if shares_wanted <= 0:
            return build_validation_result(
                False,
                "shares",
                "The amount of shares should be greater than 0, "
                "please provide a correct number of shares to share your investment.",
            )

    # Validate if users enter the date from now to the future
    if start_date is not None:
        start_date = dt.strptime(start_date, "%Y-%m-%d")
        end_date = dt.strptime(start_date, "%Y-%m-%d")
        
        if start_date < datetime.start_date.today():
            return build_validation_result(
                False,
                "start_date",
                "The start date cannot be in the past.",
            )
        


# In[ ]:


### Intents handlers ###

def company_request_handler(intent_request):

    # Get slots values from users

    company_request = get_slots(intent_request)['company']  ###the slot name
    investment_amount = get_slots(intent_request)["investment_amount"]
    shares_wanted = get_slots(intent_request)['shares']
    start_date = get_slots(intent_request)['start_date']
    end_date = get_slots(intent_request)['end_date']
    
    # Gets the invocation source, for Lex dialogs "DialogCodeHook" is expected.
    source = intent_request["invocationSource"]  #

    if source == "DialogCodeHook":
        # This code performs basic validation on the supplied input slots.

        # Gets all the slots
        slots = get_slots(intent_request)
        
        
    """
    So far, these are for validating inputs data for investment
    """

        # Validates user's input using the validate_data function
    validation_result = validate_data(investment_amount, shares_wanted, start_date, end_date)

        # If the data provided by the user is not valid,
        # the elicitSlot dialog action is used to re-prompt for the first violation detected.
        
    if not validation_result["isValid"]:
        slots[validation_result["violatedSlot"]] = None  # Cleans invalid slot

            # Returns an elicitSlot dialog to request new data for the invalid slot
            
        return elicit_slot(
            intent_request["sessionAttributes"],
            intent_request["currentIntent"]["name"],
            slots,
            validation_result["violatedSlot"],
            validation_result["message"],
        )

        # Fetch current session attributes
        output_session_attributes = intent_request["sessionAttributes"]

        # Once all slots are valid, a delegate dialog is returned to Lex to choose the next course of action.
        return delegate(output_session_attributes, get_slots(intent_request))
   
    
    """
    Keep adding other our output messages here
    
    Do not know how it works based on the order. Do we need to pass these output as slots?
    """
    
    
    # Get the current price of stock in USD.
           
    stock_price = last_close_price()
    portfolio_return = signals_df.iloc[end_date]['Portfolio Total']
    
    # Return a message with stock most recent close price.
    return close(
        intent_request["sessionAttributes"],
        "Fulfilled",
        {
            "contentType": "PlainText",
            "content": """Thank you for your information; with an investment of {} in the company {}
            starting on {}. Had you invested with us and our Stocky algorithm, you portfolio would be
            worth {} on the indicated end date of {}.
            """.format(
                investment_amount, company_request, start_date,  stock_price, end_date
            ),
        },
    )


# In[ ]:


### Intents Dispatcher ###

def dispatch(intent_request):
    
    """
    Called when the user specifies an intent for this bot.
    """

    # Get the name of the current intent
    intent_name = intent_request["currentIntent"]["name"]

    # Dispatch to bot's intent handlers
    if intent_name == "ChatStockyToMe":
        return company_request_handler(intent_request)

    raise Exception("Intent with name " + intent_name + " not supported")


# In[ ]:


### Main Handler ###

def trading_handler(event, context):
    
    """
    Route the incoming request based on intent.
    The JSON body of the request is provided in the event slot.
    """

    return dispatch(event)


# In[ ]:




