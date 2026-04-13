import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime, timedelta

def collect_historical_prices_with_features(ticker, start_date, end_date, output_format):
    print(f"Downloading historical prices for {ticker} from {start_date} to {end_date}...")
    
    # 1. Download the daily historical data
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return
        
    df = df.reset_index()
    
    # Flatten the multi-level columns
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Rename the columns to match your format
    df.rename(columns={
        'Date': 'datetime', 
        'Open': 'start_price', 
        'Close': 'end_price',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    }, inplace=True)
    
    if 'Adj Close' in df.columns:
        df.drop(columns=['Adj Close'], inplace=True)
        
    # ---------------------------------------------------------
    # PART A: CALCULATE PRE-MARKET FEATURES (Daily based)
    # ---------------------------------------------------------
    print("Calculating Pre-Market Features...")
    
    # Feature 1: Pre-Market Gap Percentage
    df['yesterday_close'] = df['end_price'].shift(1)
    df['pre_market_gap_pct'] = (df['start_price'] - df['yesterday_close']) / df['yesterday_close']
    
    # Feature 3: Gap Direction Matches Trend
    df['SMA_5'] = df['end_price'].rolling(window=5).mean()
    df['yesterday_sma'] = df['SMA_5'].shift(1)
    df['day_before_yesterday_sma'] = df['SMA_5'].shift(2)
    
    trend_is_up = df['yesterday_sma'] > df['day_before_yesterday_sma']
    gap_is_up = df['pre_market_gap_pct'] > 0
    df['gap_direction_matches_trend'] = (trend_is_up == gap_is_up).astype(int)
    
    df.drop(columns=['yesterday_close', 'SMA_5', 'yesterday_sma', 'day_before_yesterday_sma'], inplace=True)

    # ---------------------------------------------------------
    # PART A: CALCULATE PRE-MARKET VOLUME (Intraday based)
    # ---------------------------------------------------------
    print("Fetching intraday data for pre-market volume (Note: yfinance limits this to the last 730 days)...")
    try:
        # Calculate the absolute oldest date yfinance will accept for 1h data
        max_lookback_date = datetime.now() - timedelta(days=729)
        user_start_date = pd.to_datetime(start_date)
        
        # If requested start date is too old, adjust it just for the intraday pull
        if user_start_date < max_lookback_date:
            intraday_start = max_lookback_date.strftime('%Y-%m-%d')
            print(f"  -> Start date too old for intraday. Adjusting intraday start to {intraday_start}.")
        else:
            intraday_start = start_date

        intraday = yf.download(ticker, start=intraday_start, end=end_date, interval="1h", prepost=True, progress=False)
        
        if not intraday.empty:
            intraday = intraday.reset_index()
            intraday.columns = [col[0] if isinstance(col, tuple) else col for col in intraday.columns]
            
            # Ensure timezone mapping
            if intraday['Datetime'].dt.tz is None:
                intraday['Datetime'] = intraday['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            else:
                intraday['Datetime'] = intraday['Datetime'].dt.tz_convert('US/Eastern')
                
            intraday.set_index('Datetime', inplace=True)
            
            # Filter for pre-market hours (4:00 AM to 9:30 AM)
            pm_data = intraday.between_time('04:00', '09:29')
            
            # Group by date and sum the volume
            pm_data['date_only'] = pm_data.index.date
            pm_volume_daily = pm_data.groupby('date_only')['Volume'].sum().reset_index()
            pm_volume_daily.rename(columns={'date_only': 'date_match', 'Volume': 'pm_volume'}, inplace=True)
            pm_volume_daily['date_match'] = pd.to_datetime(pm_volume_daily['date_match'])
            
            # Merge with our main dataframe
            df['date_match'] = pd.to_datetime(df['datetime']).dt.normalize()
            df = df.merge(pm_volume_daily, on='date_match', how='left')
            
            # Feature 2: Relative Pre-Market Volume
            df['avg_10d_pm_volume'] = df['pm_volume'].rolling(window=10).mean().shift(1)
            df['pre_market_volume_relative'] = df['pm_volume'] / df['avg_10d_pm_volume']
            
            df.drop(columns=['date_match', 'pm_volume', 'avg_10d_pm_volume'], inplace=True)
        else:
            df['pre_market_volume_relative'] = np.nan
    except Exception as e:
        print(f"Could not fetch intraday data: {e}")
        df['pre_market_volume_relative'] = np.nan

    # ---------------------------------------------------------
    # FINAL CLEANUP & EXPORT
    # ---------------------------------------------------------
    # Impute missing pre-market volume data with 1.0 (representing 'average' volume)
    df['pre_market_volume_relative'] = df['pre_market_volume_relative'].fillna(1.0)
    
    # Format datetime
    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Replace Pandas NaNs with None so JSON dumps it as valid 'null' (mostly for any other missing rolling metrics)
    df = df.replace({np.nan: None})
    
    os.makedirs("data/raw", exist_ok=True)
    
    if output_format.lower() == 'json':
        file_path = f"data/raw/{ticker}_prices_features_{start_date}_to_{end_date}.json"
        records = df.to_dict(orient='records')
        
        with open(file_path, 'w') as f:
            f.write("[\n")
            for i, record in enumerate(records):
                json_str = json.dumps(record)
                if i < len(records) - 1:
                    f.write(f"  {json_str},\n")
                else:
                    f.write(f"  {json_str}\n")
            f.write("]\n")
            
    else:
        file_path = f"data/raw/{ticker}_prices_features_{start_date}_to_{end_date}.csv"
        df.to_csv(file_path, index=False)
        
    print(f"Successfully saved {len(df)} days of historical data with features to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical stock prices with pre-market features.")
    
    parser.add_argument("--ticker", type=str, default="NVDA", help="The stock ticker symbol (e.g., NVDA)")
    parser.add_argument("--start", type=str, required=True, help="Entry/Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Exit/End date (YYYY-MM-DD)")
    parser.add_argument("--format", type=str, choices=['csv', 'json'], default='json', help="Output format: csv or json")
    
    args = parser.parse_args()
    
    collect_historical_prices_with_features(args.ticker, args.start, args.end, args.format)