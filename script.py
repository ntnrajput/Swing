import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def merge_levels_up(array, tolerance_percentage=5):
    
    array = array.to_numpy()
    sorted_array = sorted(array)
    sorted_array.sort(reverse=True)
    result = [sorted_array[0]]  # Initialize result with the first element
 
    for i in range(1, len(sorted_array)):
        current_number = sorted_array[i]
        previous_number = result[-1]

        # Check if the current number is more than 5% greater than the previous number
        if current_number < previous_number * (1 - tolerance_percentage / 100):
            result.append(current_number)
    return result

def merge_levels_down(array, tolerance_percentage=5):
    
    array = array.to_numpy()
    sorted_array = sorted(array)
    sorted_array.sort()
    result = [sorted_array[0]]  # Initialize result with the first element
 
    for i in range(1, len(sorted_array)):
        current_number = sorted_array[i]
        previous_number = result[-1]

        # Check if the current number is more than 5% greater than the previous number
        if current_number > previous_number * (1 + tolerance_percentage / 100):
            result.append(current_number)
    return result



def get_max (reversal_points):
    first_row = reversal_points.iloc[0]  # First row from slope_change_points
    first_date = first_row['Date']
    first_close = first_row['Close']

    # Append the first value to df_maximums
    df_maximums=pd.DataFrame({"Date":[first_date],"Maximums":[first_close]})

    # Initialize variables for maximum and minimum
    max_value = itc_data.loc[0, 'Close']
    min_value = itc_data.loc[0, 'Close']


    for index, row in reversal_points.iterrows():
        
        if row['Change'] =="Increased":
            last_max = df_maximums['Maximums'].iloc[-1]
            if((row['Close'] > (1.05 * last_max)) or ((row['Close'] < (0.95 * last_max)) ) ):
                df_maximums.loc[len(df_maximums.index)] = [row['Date'],row['Close']] 
            else:
                if(row['Close']>last_max):                  
                    df_maximums = df_maximums[:-1]
                    df_maximums.loc[len(df_maximums.index)] = [row['Date'],row['Close']] 
    
    return df_maximums
    
def get_min (reversal_points):
    first_row = reversal_points.iloc[0]  # First row from slope_change_points
    first_date = first_row['Date']
    first_close = first_row['Close']

    # Append the first value to df_maximums
    df_minimums=pd.DataFrame({"Date":[first_date],"Minimums":[first_close]})

    for index, row in reversal_points.iterrows():
        if row['Change'] =="Decreased":
            last_min = df_minimums['Minimums'].iloc[-1]
            if((row['Close'] > (1.05 * last_min)) or ((row['Close'] < (0.95 * last_min)) ) ):
                df_minimums.loc[len(df_minimums.index)] = [row['Date'],row['Close']] 
            else:
                if(row['Close']<last_min):                  
                    df_minimums = df_minimums[:-1]
                    df_minimums.loc[len(df_minimums.index)] = [row['Date'],row['Close']] 
    
    return df_minimums

def check_level_crossing(imp_levels_max,current_price,previous_day_price,symbol):
    print(symbol)
    for levels in imp_levels_max:
        if (previous_day_price < levels) and (current_price > levels):
            print("time to Buy")

# Define the list of Nifty 200 stock symbols
nifty_200_symbols = ['ITC.NS', 'TITAN.NS','TECHM.NS','RITES.NS','ULTRACEMCO.NS','MARUTI.NS','BAJFINANCE.NS','COALINDIA.NS','APOLLOHOSP.NS',
                     'HDFCLIFE.NS',]  # Replace with actual symbols

# Specify the date range for the historical data (5 years ago from today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')

# Create an empty DataFrame to store the data
nifty_200_data = pd.DataFrame()

# Fetch historical data for each stock
for symbol in nifty_200_symbols:
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        stock_data = stock_data.reset_index()  # Reset index to have 'Date' as a column
        stock_data['Symbol'] = symbol  # Add a column for stock symbol
        stock_data = stock_data[['Symbol', 'Date', 'Close', 'Volume']]
        nifty_200_data = pd.concat([nifty_200_data, stock_data])
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

for symbol in nifty_200_symbols:
    # Filter data for ITC
    itc_data = nifty_200_data[nifty_200_data['Symbol'] == symbol]
    current_price = itc_data.iloc[-1]['Close']

    itc_data = itc_data[:-1]
    previous_day_price = itc_data.iloc[-1]['Close']
    itc_data['Serial Number'] = range(1, len(itc_data) + 1)



    itc_data['Change'] = itc_data['Close'].diff().apply(lambda x: 'Increased' if x > 0 else 'Decreased')
    itc_data['Sign Change'] = itc_data['Change'] != itc_data['Change'].shift(1)

    # Create a new DataFrame with only points where slope changes sign
    slope_change_points = itc_data[itc_data['Sign Change']]
    rows_with_sign_change = itc_data[itc_data['Sign Change']].index.to_numpy()

    rows_with_sign_change[1:] -= 1

    reversal_points = itc_data.loc[rows_with_sign_change]

    df_maximums = get_max(reversal_points)
    df_minimums = get_min(reversal_points)

    imp_levels_max = merge_levels_up(df_maximums['Maximums'])
    imp_levels_min = merge_levels_down(df_minimums['Minimums'])

    # plt.figure(figsize=(10, 6))
    # plt.plot(df_maximums['Date'], df_maximums['Maximums'], marker='o', label='Maximums')
    # plt.plot(df_minimums['Date'], df_minimums['Minimums'], marker='x', label='Minimums')
    # plt.plot(itc_data['Date'], itc_data['Close'], marker='', label='Close')
    # for level in imp_levels_max:
    #     plt.axhline(y=level, linestyle='--', label=f'Level {level}')
    # for level in imp_levels_min:
    #     plt.axhline(y=level, linestyle=':', label=f'Level {level}')

    # plt.title('Graph of Maximums Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Maximums')
    # # plt.legend()
    # plt.grid(True)
    # plt.show()
    print(symbol)
    if current_price > previous_day_price:
        check_level_crossing(imp_levels_max,current_price,previous_day_price,symbol)




    