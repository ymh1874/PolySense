import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime, timedelta

def collect_historical_prices_with_features(ticker, start_date, end_date, output_format):
    print(f"Downloading historical prices for {ticker} from {start_date} to {end_date}...")
    
    # Download the standard daily price history from Yahoo Finance.
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Gracefully exit if the API returns no data for the requested date range.
    if df.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return
        
    # Convert the date index into a standard column so we can manipulate it easily.
    df = df.reset_index()
    
    # yfinance often returns multi-level column headers (e.g., ('Close', 'NVDA')). 
    # This flattens them to a single level (e.g., 'Close') to prevent indexing errors.
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Standardize column names to match your required dataset schema.
    df.rename(columns={
        'Date': 'datetime', 
        'Open': 'start_price', 
        'Close': 'end_price',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    }, inplace=True)
    
    # Remove the 'Adj Close' column as we are working with raw daily closes.
    if 'Adj Close' in df.columns:
        df.drop(columns=['Adj Close'], inplace=True)
        
    print("Calculating Pre-Market Features...")
    
    # Feature 1: Pre-Market Gap Percentage
    # We shift the end_price down by 1 row to get yesterday's close, then calculate the % difference 
    # between today's open (start_price) and yesterday's close.
    df['yesterday_close'] = df['end_price'].shift(1)
    df['pre_market_gap_pct'] = (df['start_price'] - df['yesterday_close']) / df['yesterday_close']
    
    # Feature 3: Gap Direction Matches Trend
    # Calculate a 5-day Simple Moving Average (SMA) to represent the short-term trend.
    df['SMA_5'] = df['end_price'].rolling(window=5).mean()
    # Compare yesterday's SMA to the day before to determine if the 5-day trend is sloping up or down.
    df['yesterday_sma'] = df['SMA_5'].shift(1)
    df['day_before_yesterday_sma'] = df['SMA_5'].shift(2)
    
    # Create binary labels (True/False) for whether the trend is up and whether the overnight gap was up.
    trend_is_up = df['yesterday_sma'] > df['day_before_yesterday_sma']
    gap_is_up = df['pre_market_gap_pct'] > 0
    # Cast the boolean comparison to an integer (1 if they match, 0 if they contradict).
    df['gap_direction_matches_trend'] = (trend_is_up == gap_is_up).astype(int)
    
    # Clean up temporary calculation columns to keep the dataframe tidy.
    df.drop(columns=['yesterday_close', 'SMA_5', 'yesterday_sma', 'day_before_yesterday_sma'], inplace=True)

    print("Fetching intraday data for pre-market volume (Note: yfinance limits this to the last 730 days)...")
    try:
        # yfinance strictly enforces a 730-day limit on granular 1-hour data.
        # We calculate the absolute oldest allowed date.
        max_lookback_date = datetime.now() - timedelta(days=729)
        user_start_date = pd.to_datetime(start_date)
        
        # If the user asks for 5 years of data, we adjust the intraday start date to the maximum 
        # 730 days allowed, preventing the entire API call from failing.
        if user_start_date < max_lookback_date:
            intraday_start = max_lookback_date.strftime('%Y-%m-%d')
            print(f"  -> Start date too old for intraday. Adjusting intraday start to {intraday_start}.")
        else:
            intraday_start = start_date

        # Fetch the 1-hour data. prepost=True ensures we get Extended Trading Hours (pre-market/after-hours).
        intraday = yf.download(ticker, start=intraday_start, end=end_date, interval="1h", prepost=True, progress=False)
        
        if not intraday.empty:
            intraday = intraday.reset_index()
            intraday.columns = [col[0] if isinstance(col, tuple) else col for col in intraday.columns]
            
            # Timezone mapping: Ensure the timestamps are strictly set to US/Eastern time so 
            # our 4:00 AM to 9:30 AM filter aligns perfectly with real-world market hours.
            if intraday['Datetime'].dt.tz is None:
                intraday['Datetime'] = intraday['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            else:
                intraday['Datetime'] = intraday['Datetime'].dt.tz_convert('US/Eastern')
                
            intraday.set_index('Datetime', inplace=True)
            
            # Filter the dataset down to purely pre-market hours.
            pm_data = intraday.between_time('04:00', '09:29')
            
            # Aggregate the 1-hour chunks into a single total pre-market volume sum per day.
            pm_data['date_only'] = pm_data.index.date
            pm_volume_daily = pm_data.groupby('date_only')['Volume'].sum().reset_index()
            pm_volume_daily.rename(columns={'date_only': 'date_match', 'Volume': 'pm_volume'}, inplace=True)
            pm_volume_daily['date_match'] = pd.to_datetime(pm_volume_daily['date_match'])
            
            # Normalize the daily timeframe's datetime to 00:00:00 so we can safely merge it with the intraday aggregate.
            df['date_match'] = pd.to_datetime(df['datetime']).dt.normalize()
            df = df.merge(pm_volume_daily, on='date_match', how='left')
            
            # Feature 2: Relative Pre-Market Volume
            # Calculate the 10-day rolling average of pre-market volume. min_periods=1 ensures 
            # we don't drop the first 9 days of our dataset; it just averages whatever is available.
            df['avg_10d_pm_volume'] = df['pm_volume'].rolling(window=10, min_periods=1).mean().shift(1)
            # Divide today's pre-market volume by the 10-day average to see if activity is abnormally high/low.
            df['pre_market_volume_relative'] = df['pm_volume'] / df['avg_10d_pm_volume']
            
            # Drop the temporary join keys and raw volume columns.
            df.drop(columns=['date_match', 'pm_volume', 'avg_10d_pm_volume'], inplace=True)
        else:
            df['pre_market_volume_relative'] = np.nan
    except Exception as e:
        print(f"Could not fetch intraday data: {e}")
        df['pre_market_volume_relative'] = np.nan

    # Fill any missing relative pre-market volume values (e.g., from days older than 730 days) 
    # with 1.0, representing a standard "average" day to prevent ML models from crashing on NaNs.
    df['pre_market_volume_relative'] = df['pre_market_volume_relative'].fillna(1.0)
    
    # Format the primary datetime column into standard ISO 8601 string format.
    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Replace Pandas specific NaN/NaT objects with standard Python None objects. 
    # This guarantees they will be parsed as 'null' in the final JSON.
    df = df.replace({np.nan: None})
    
    # Ensure the output directory exists before attempting to save the file.
    os.makedirs("data/raw", exist_ok=True)
    
    # Export the final dataset using pandas built-in functions.
    if output_format.lower() == 'json':
        file_path = f"data/raw/{ticker}_prices_features_{start_date}_to_{end_date}.json"
        # to_json with orient='records' automatically creates a clean list of JSON objects, and indent handles formatting.
        df.to_json(file_path, orient='records', indent=2)
    else:
        file_path = f"data/raw/{ticker}_prices_features_{start_date}_to_{end_date}.csv"
        df.to_csv(file_path, index=False)
        
    print(f"Successfully saved {len(df)} days of historical data with features to {file_path}")

# Command-Line Interface setup block.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical stock prices with pre-market features.")
    
    # Define the flags users can pass from the terminal.
    parser.add_argument("--ticker", type=str, default="NVDA", help="The stock ticker symbol (e.g., NVDA)")
    parser.add_argument("--start", type=str, required=True, help="Entry/Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Exit/End date (YYYY-MM-DD)")
    parser.add_argument("--format", type=str, choices=['csv', 'json'], default='json', help="Output format: csv or json")
    
    args = parser.parse_args()
    
    # Trigger the main function using the parsed terminal arguments.
    collect_historical_prices_with_features(args.ticker, args.start, args.end, args.format)