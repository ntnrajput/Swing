import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

print('Automatic')
Stocks = []

def send_email(subject, body):
    # Email configurations
    sender_email = "criobd.rites@gmail.com"
    sender_password = "eqhl qvmp ruaz jrlk"
    receiver_email = "ntn.rajput89@gmail.com"

    # Create the MIME object
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach the body of the email
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Setup the server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        # Login to your Gmail account
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, receiver_email, msg.as_string())

        # Quit the server
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def merge_levels_up(array, tolerance_percentage=3):
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

                      last_index = df_maximums.index[-1]
                      df_maximums.iloc[last_index] = [row['Date'], row['Close']]
                      # df_maximums_new = df_maximums[:-1].copy()
                      # df_maximums_new.loc[len(df_maximums_new.index)] = [row['Date'],row['Close']]

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


def check_level_crossing(imp_levels_max,current_price,previous_day_price,parso_price,symbol,all_high,ma_20,ma_50):
    global Stocks

    for i in range (len(imp_levels_max)-1):
      near_high = 0
      levels = imp_levels_max[i]
      if i > 0 and i< (len(imp_levels_max)-1) :
        nxt_level = imp_levels_max[i-1]
        lower_level = imp_levels_max[i+1]
      # if i == (len(imp_levels_max)-1):
      if(i==0):
        near_high = 1
        lower_level = imp_levels_max[i+1]


      if ((previous_day_price <  levels) and (current_price > levels) ) or ((parso_price > previous_day_price) and (previous_day_price < 1.01* levels) and (current_price > levels)):

        if(current_price < 1.05 * ma_20 ) and (ma_20 > 1.10 * ma_50) and (levels < 1.01 * ma_20):
            print("time to Buy ma 20", symbol, 'for crossing', levels)
            print('symbol',symbol,parso_price,previous_day_price,current_price)
            print(imp_levels_max)
            Stocks.append([symbol,"ma 20",levels])

        if (near_high == 1) and (current_price < 1.02 * levels)  :
            print("time to Buy  near high", symbol, 'for crossing', levels)
            print(imp_levels_max)
            Stocks.append([symbol,"High",levels])

        if (near_high == 0 ) and (nxt_level > 1.07*levels) and (lower_level > 0.95 * levels) and (nxt_level > 1.05 * current_price):
            print("time to Buy", symbol, 'for crossing', levels, 'next level:',nxt_level)
            print(imp_levels_max)
            Stocks.append([symbol,levels])

# Define the list of Nifty 200 stock symbols
nifty_200_symbols = ['ITC.NS', 'TITAN.NS','TECHM.NS','RITES.NS','ULTRACEMCO.NS','MARUTI.NS','BAJFINANCE.NS','COALINDIA.NS','APOLLOHOSP.NS',
                     'HDFCLIFE.NS','MAHSEAMLES.NS','RELIANCE.NS','TCS.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','BHARTIARTL.NS','HINDUNILVR.NS',
                     'ITC.NS','SBIN.NS','LICI.NS','LT.NS','HCLTECH.NS','KOTAKBANK.NS','ADANIENT.NS','AXISBANK.NS','SUNPHARMA.NS','ASIANPAINT.NS','NTPC.NS',
                     'TATAMOTORS.NS','ONGC.NS','ADANIGREEN.NS','BAJAJFINSV.NS','ADANIPORTS.NS','DMART.NS','NESTLEIND.NS','WIPRO.NS','POWERGRID.NS','ADANIPOWER.NS',
                     'BAJAJ-AUTO.NS','M&M.NS','JSWSTEEL.NS','HAL.NS','DLF.NS','IOC.NS','LTIM.NS','TATASTEEL.NS','VBL.NS','JIOFIN.NS','IRFC.NS','SIEMENS.NS','SBILIFE.NS',
                     'GRASIM.NS','PIDILITIND.NS','HDFCLIFE.NS','BEL.NS','HINDZINC.NS','PFC.NS','INDUSINDBK.NS','ADANIENSOL.NS','BRITANNIA.NS','HINDALCO.NS','ZOMATO.NS','BANKBARODA.NS',
                     'INDIGO.NS','GODREJCP.NS','LODHA.NS','ATGL.NS','TATAPOWER.NS','TRENT.NS','RECLTD.NS','TATACONSUM.NS','PNB.NS','CIPLA.NS','AMBUJACEM.NS','GAIL.NS',
                     'EICHERMOT.NS','CHOLAFIN.NS','DIVISLAB.NS','ABB.NS','VEDL.NS','TVSMOTOR.NS','BPCL.NS','DABUR.NS','UNIONBANK.NS','SHREECEM.NS','DRREDDY.NS','HAVELLS.NS',
                     'ABB.BO','ALKEM.BO','ASHOKLEY.BO','AUBANK.BO','AUROPHARMA.BO','BAJAJHLDNG.BO','BALKRISHNA.BO','BANKINDIA.BO','BERGEPAINT.BO','BHARATFORG.BO',
                     'BHEL.BO','BOSCHLTD.BO','BPCL.BO','CANBK.BO','CGPOWER.BO','CHOLAFIN.BO','CIPLA.BO','COLPAL.BO','CONCOR.BO','CUMMINSIND.BO','DIVISLAB.BO','FACT.BO',
                     'DRREDDY.BO','EICHERMOT.BO','GICRE.BO','GMRINFRA.BO','HDFCAMC.BO','GODREJPROP.BO','HEROMOTOCO.BO','HINDALCO.BO','HINDPETRO.BO','HINDUNILVR.BO','HINDZINC.BO',
                     'ICICIBANK.BO','ICICIGI.BO','IDBI.BO','IDEA.BO','IDFCFIRSTB.BO','INDHOTEL.BO','INDIANB.BO','INDIGO.BO','INDUSINDBK.BO','INDUSTOWER.BO','IOB.BO','IRCTC.BO',
                     'IRFC.BO','JINDALSTEL.BO','JIOFIN.BO','JSWENERGY.BO','JSWSTEEL.BO','KOTAKBANK.BO','LICI.BO','LODHA.BO','LT.BO','LTIM.BO','LTTS.BO','LUPIN.BO','M&M.BO','MANKIND.BO',
                     'MARICO.BO','MAXHEALTH.BO','MOTHERSON.BO','MRF.BO','MUTHOOTFIN.BO','NAUKRI.BO','NESTLEIND.BO','NHPC.BO','NMDC.BO','NTPC.BO','OBEROIRLTY.BO','OFSS.BO','ONGC.BO']  # Replace with actual symbols

# nifty_200_symbols = ['ITC.NS', 'TITAN.NS','TECHM.NS','RITES.NS']

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
        stock_data = stock_data[['Symbol', 'Date', 'Close', 'Open']]
        nifty_200_data = pd.concat([nifty_200_data, stock_data])
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")



for symbol in nifty_200_symbols:
    # Filter data for ITC
    itc_data = nifty_200_data[nifty_200_data['Symbol'] == symbol]
    share_details = yf.Ticker(symbol)
    share_history = share_details.history()
    current_price = share_history['Close'].iloc[-1]
    open_price = share_history['Open'].iloc[-1]
    previous_day_price = itc_data.iloc[-1]['Close']
    parso_price = itc_data.iloc[-2]['Close']
    all_high = max(itc_data['Close'])


    if current_price ==previous_day_price:
      previous_day_price = itc_data.iloc[-2]['Close']
      parso_price = itc_data.iloc[-3]['Close']
      open_price = itc_data.iloc[-1]['Open']


    ma_20= itc_data['Close'].tail(20).mean()
    ma_50= itc_data['Close'].tail(50).mean()

    itc_data = itc_data.copy()
    itc_data['Change'] = itc_data['Close'].diff().apply(lambda x: 'Increased' if x > 0 else 'Decreased')

    itc_data['Sign Change'] = itc_data['Change'] != itc_data['Change'].shift(1)


    delta = current_price - open_price

    # Create a new DataFrame with only points where slope changes sign
    slope_change_points = itc_data[itc_data['Sign Change']]

    rows_with_sign_change = itc_data[itc_data['Sign Change']].index.to_numpy()

    rows_with_sign_change[1:] -= 1

    reversal_points = itc_data.iloc[rows_with_sign_change]
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
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # print(symbol,parso_price, previous_day_price,current_price)

    print(symbol, open_price, current_price, delta)

    if current_price > previous_day_price and current_price > ma_20 and ma_20 > ma_50 and delta > 0:
      check_level_crossing(imp_levels_max,current_price,previous_day_price,parso_price,symbol,all_high,ma_20, ma_50)


Shares = ' & '.join([' '.join(map(str, inner_list)) for inner_list in Stocks])
send_email('Swing Tip', Shares)
